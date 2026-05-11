/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/



#include "System.h"
#include "Converter.h"

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <pangolin/pangolin.h>

namespace ORB_SLAM2
{

namespace
{
string JoinPath(const string &a, const string &b)
{
    if(a.empty())
        return b;
    if(a[a.size()-1]=='/')
        return a + b;
    return a + "/" + b;
}

string Basename(const string &path)
{
    const size_t pos = path.find_last_of("/\\");
    if(pos==string::npos)
        return path;
    return path.substr(pos+1);
}

bool EnsureDirectory(const string &path)
{
    if(path.empty())
        return false;

    if(path=="/")
        return true;

    string current;
    if(path[0]=='/')
        current = "/";

    stringstream ss(path);
    string segment;
    while(getline(ss, segment, '/'))
    {
        if(segment.empty() || segment==".")
            continue;

        if(!current.empty() && current[current.size()-1]!='/')
            current += "/";
        current += segment;

        if(mkdir(current.c_str(), 0755)!=0 && errno!=EEXIST)
            return false;
    }

    return true;
}

bool IsFinitePoint3f(const cv::Mat &pos)
{
    if(pos.empty() || pos.rows<3)
        return false;
    return std::isfinite(pos.at<float>(0)) &&
           std::isfinite(pos.at<float>(1)) &&
           std::isfinite(pos.at<float>(2));
}

bool HasNonZeroDistortion(const cv::Mat &distCoef, const double eps = 1e-12)
{
    if(distCoef.empty())
        return false;

    for(int r=0; r<distCoef.rows; r++)
    {
        for(int c=0; c<distCoef.cols; c++)
        {
            if(std::fabs(distCoef.at<double>(r,c)) > eps)
                return true;
        }
    }
    return false;
}

bool SaveImageWithOptionalUndistortion(const string &srcImagePath,
                                       const string &dstImagePath,
                                       const cv::Mat &K,
                                       const cv::Mat &distCoef,
                                       const bool undistortImage)
{
    cv::Mat image = cv::imread(srcImagePath, CV_LOAD_IMAGE_UNCHANGED);
    if(image.empty())
        return false;

    cv::Mat imageToWrite = image;
    if(undistortImage)
    {
        cv::Mat undistorted;
        cv::undistort(image, undistorted, K, distCoef);
        if(undistorted.empty())
            return false;
        imageToWrite = undistorted;
    }

    return cv::imwrite(dstImagePath, imageToWrite);
}
} // namespace

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)),
        mptViewer(static_cast<std::thread*>(NULL)), mbReset(false), mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize Tracking. It runs in the thread that calls Track*().
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize Local Mapping. It is advanced explicitly after each Tracking step.
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR, strSettingsFile);
    mpLocalMapper->SetSingleThreadedMode(true);

    //Initialize Loop Closing. It is advanced explicitly after each Local Mapping step.
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mpLoopCloser->SetSingleThreadedMode(true);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between SLAM modules.
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);
    RunSingleThreadedBackEnd();

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);
    RunSingleThreadedBackEnd();

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);
    RunSingleThreadedBackEnd();

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::RunSingleThreadedBackEnd()
{
    mpLocalMapper->RunOnce();
    mpLoopCloser->RunOnce();
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveMapToCOLMAP(const string &strOutputDir,
                             const std::vector<string> &vstrImageFilenames,
                             const string &strSettingsFile)
{
    cout << endl << "Saving final map as COLMAP text model to " << strOutputDir << " ..." << endl;

    if(vstrImageFilenames.empty())
    {
        cerr << "ERROR: Image filename list is empty. Cannot export keyframe images." << endl;
        return;
    }

    const string sparseDir = JoinPath(JoinPath(strOutputDir, "sparse"), "0");
    const string imageDir = JoinPath(strOutputDir, "images");
    if(!EnsureDirectory(strOutputDir) || !EnsureDirectory(sparseDir) || !EnsureDirectory(imageDir))
    {
        cerr << "ERROR: Could not create export directories under: " << strOutputDir << endl;
        return;
    }

    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Failed to open settings file at: " << strSettingsFile << endl;
        return;
    }

    const double fx = static_cast<double>(fsSettings["Camera.fx"]);
    const double fy = static_cast<double>(fsSettings["Camera.fy"]);
    const double cx = static_cast<double>(fsSettings["Camera.cx"]);
    const double cy = static_cast<double>(fsSettings["Camera.cy"]);
    const double k1 = static_cast<double>(fsSettings["Camera.k1"]);
    const double k2 = static_cast<double>(fsSettings["Camera.k2"]);
    const double p1 = static_cast<double>(fsSettings["Camera.p1"]);
    const double p2 = static_cast<double>(fsSettings["Camera.p2"]);
    double k3 = 0.0;
    if(!fsSettings["Camera.k3"].empty())
        k3 = static_cast<double>(fsSettings["Camera.k3"]);

    if(fx==0.0 || fy==0.0)
    {
        cerr << "ERROR: Invalid camera intrinsics in settings file." << endl;
        return;
    }

    const cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0.0, cx,
                                                0.0, fy, cy,
                                                0.0, 0.0, 1.0);
    const cv::Mat distCoef = (cv::Mat_<double>(1,5) << k1, k2, p1, p2, k3);
    const bool bUndistortExportImages = HasNonZeroDistortion(distCoef);

    struct ExportedImage
    {
        int imageId;
        unsigned long frameId;
        string imageName;
        string srcImagePath;
        cv::Mat Tcw;
        vector<cv::KeyPoint> vKeysUn;
        vector<MapPoint*> vMPMatches;
        vector<bool> vbOutlier;
    };

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    vector<ExportedImage> vExportedImages;
    set<unsigned long> sExportedFrameIds;
    int nextImageId = 1;
    int nExportedMapKFs = 0;
    int nExportedLocalizationKFs = 0;

    auto AddExportedImage = [&](const unsigned long frameId,
                                const string &prefix,
                                const cv::Mat &Tcw,
                                const vector<cv::KeyPoint> &vKeysUn,
                                const vector<MapPoint*> &vMPMatches,
                                const vector<bool> &vbOutlier) -> bool
    {
        if(frameId >= vstrImageFilenames.size())
            return false;
        if(sExportedFrameIds.count(frameId))
            return false;
        if(Tcw.empty())
            return false;

        const string srcImagePath = vstrImageFilenames[frameId];
        if(srcImagePath.empty())
            return false;

        stringstream ss;
        ss << prefix << "_" << setfill('0') << setw(6) << frameId << "_" << Basename(srcImagePath);
        const string imageName = ss.str();
        const string dstImagePath = JoinPath(imageDir, imageName);
        if(!SaveImageWithOptionalUndistortion(srcImagePath, dstImagePath, K, distCoef, bUndistortExportImages))
        {
            cerr << "WARNING: Could not save keyframe image: " << srcImagePath << endl;
            return false;
        }

        ExportedImage eimg;
        eimg.imageId = nextImageId++;
        eimg.frameId = frameId;
        eimg.imageName = imageName;
        eimg.srcImagePath = srcImagePath;
        eimg.Tcw = Tcw.clone();
        eimg.vKeysUn = vKeysUn;
        eimg.vMPMatches = vMPMatches;
        eimg.vbOutlier = vbOutlier;
        vExportedImages.push_back(eimg);
        sExportedFrameIds.insert(frameId);
        return true;
    };

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(!pKF || pKF->isBad())
            continue;
        const vector<MapPoint*> vMPMatches = pKF->GetMapPointMatches();
        const vector<bool> vbOutlier(vMPMatches.size(), false);
        if(AddExportedImage(pKF->mnFrameId, "kf", pKF->GetPose(), pKF->mvKeysUn, vMPMatches, vbOutlier))
            nExportedMapKFs++;
    }

    const vector<Tracking::LocalizationKeyFrameSnapshot> vLocKFSnapshots =
            mpTracker->GetLocalizationKeyFrameSnapshots();
    for(size_t i=0; i<vLocKFSnapshots.size(); i++)
    {
        const Tracking::LocalizationKeyFrameSnapshot &snapshot = vLocKFSnapshots[i];
        if(AddExportedImage(snapshot.mnFrameId, "kf", snapshot.mTcw, snapshot.mvKeysUn, snapshot.mvpMapPoints, snapshot.mvbOutlier))
            nExportedLocalizationKFs++;
    }

    if(vExportedImages.empty())
    {
        cerr << "ERROR: No valid keyframes were exported." << endl;
        return;
    }

    int width = 0;
    int height = 0;
    if(!fsSettings["Camera.width"].empty())
        width = static_cast<int>(fsSettings["Camera.width"]);
    if(!fsSettings["Camera.height"].empty())
        height = static_cast<int>(fsSettings["Camera.height"]);

    if(width<=0 || height<=0)
    {
        cv::Mat im0 = cv::imread(vExportedImages[0].srcImagePath, CV_LOAD_IMAGE_UNCHANGED);
        if(!im0.empty())
        {
            width = im0.cols;
            height = im0.rows;
        }
    }

    if(width<=0 || height<=0)
    {
        cerr << "ERROR: Invalid image size. Add Camera.width/Camera.height to settings, or verify image files." << endl;
        return;
    }

    set<MapPoint*> sObservedMapPoints;
    for(size_t i=0; i<vExportedImages.size(); i++)
    {
        const ExportedImage &eimg = vExportedImages[i];
        const size_t nPts = std::min(eimg.vKeysUn.size(), eimg.vMPMatches.size());
        for(size_t idx=0; idx<nPts; idx++)
        {
            if(idx < eimg.vbOutlier.size() && eimg.vbOutlier[idx])
                continue;
            MapPoint* pMP = eimg.vMPMatches[idx];
            if(!pMP || pMP->isBad())
                continue;
            cv::Mat pos = pMP->GetWorldPos();
            if(!IsFinitePoint3f(pos))
                continue;
            sObservedMapPoints.insert(pMP);
        }
    }

    vector<MapPoint*> vpObservedMapPoints(sObservedMapPoints.begin(), sObservedMapPoints.end());
    sort(vpObservedMapPoints.begin(), vpObservedMapPoints.end(),
         [](MapPoint* a, MapPoint* b){ return a->mnId < b->mnId; });

    map<MapPoint*, long unsigned int> mMPToPoint3DId;
    long unsigned int nextPoint3DId = 1;
    for(size_t i=0; i<vpObservedMapPoints.size(); i++)
        mMPToPoint3DId[vpObservedMapPoints[i]] = nextPoint3DId++;

    ofstream fCameras(JoinPath(sparseDir, "cameras.txt").c_str());
    if(!fCameras.is_open())
    {
        cerr << "ERROR: Could not open cameras.txt for writing." << endl;
        return;
    }
    fCameras << "# Camera list with one line of data per camera:" << endl;
    fCameras << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << endl;
    fCameras << "# Number of cameras: 1" << endl;
    fCameras << fixed << setprecision(9);
    const double export_k1 = 0.0;
    const double export_k2 = 0.0;
    const double export_p1 = 0.0;
    const double export_p2 = 0.0;
    fCameras << 1 << " OPENCV " << width << " " << height << " "
             << fx << " " << fy << " " << cx << " " << cy << " "
             << export_k1 << " " << export_k2 << " " << export_p1 << " " << export_p2 << endl;
    fCameras.close();

    ofstream fImages(JoinPath(sparseDir, "images.txt").c_str());
    if(!fImages.is_open())
    {
        cerr << "ERROR: Could not open images.txt for writing." << endl;
        return;
    }
    fImages << "# Image list with two lines of data per image:" << endl;
    fImages << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME" << endl;
    fImages << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << endl;
    fImages << "# Number of images: " << vExportedImages.size() << endl;
    fImages << fixed << setprecision(9);

    map<MapPoint*, vector<pair<int,size_t> > > mPointTracks;
    for(size_t i=0; i<vExportedImages.size(); i++)
    {
        const ExportedImage &eimg = vExportedImages[i];
        cv::Mat Tcw = eimg.Tcw;
        cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
        cv::Mat tcw = Tcw.rowRange(0,3).col(3);
        vector<float> q = Converter::toQuaternion(Rcw);

        fImages << eimg.imageId << " "
                << q[3] << " " << q[0] << " " << q[1] << " " << q[2] << " "
                << tcw.at<float>(0) << " " << tcw.at<float>(1) << " " << tcw.at<float>(2) << " "
                << 1 << " " << eimg.imageName << endl;

        const size_t nPts = std::min(eimg.vKeysUn.size(), eimg.vMPMatches.size());
        size_t point2DIdx = 0;
        bool bFirst = true;
        for(size_t idx=0; idx<nPts; idx++)
        {
            if(idx < eimg.vbOutlier.size() && eimg.vbOutlier[idx])
                continue;
            MapPoint* pMP = eimg.vMPMatches[idx];
            if(!pMP || pMP->isBad())
                continue;
            if(!mMPToPoint3DId.count(pMP))
                continue;

            const cv::KeyPoint &kp = eimg.vKeysUn[idx];
            if(!bFirst)
                fImages << " ";
            fImages << kp.pt.x << " " << kp.pt.y << " " << mMPToPoint3DId[pMP];
            mPointTracks[pMP].push_back(make_pair(eimg.imageId, point2DIdx));
            point2DIdx++;
            bFirst = false;
        }
        fImages << endl;
    }
    fImages.close();

    ofstream fPoints(JoinPath(sparseDir, "points3D.txt").c_str());
    if(!fPoints.is_open())
    {
        cerr << "ERROR: Could not open points3D.txt for writing." << endl;
        return;
    }
    fPoints << "# 3D point list with one line of data per point:" << endl;
    fPoints << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)" << endl;
    fPoints << "# Number of points: " << mMPToPoint3DId.size() << endl;
    fPoints << fixed << setprecision(9);

    for(size_t i=0; i<vpObservedMapPoints.size(); i++)
    {
        MapPoint* pMP = vpObservedMapPoints[i];
        if(!mMPToPoint3DId.count(pMP))
            continue;
        const long unsigned int point3DId = mMPToPoint3DId[pMP];
        cv::Mat pos = pMP->GetWorldPos();

        fPoints << point3DId << " "
                << pos.at<float>(0) << " " << pos.at<float>(1) << " " << pos.at<float>(2) << " "
                << 255 << " " << 255 << " " << 255 << " " << 0.0;

        const vector<pair<int,size_t> > &tracks = mPointTracks[pMP];
        for(size_t t=0; t<tracks.size(); t++)
        {
            fPoints << " " << tracks[t].first << " " << tracks[t].second;
        }
        fPoints << endl;
    }
    fPoints.close();

    cout << "Exported " << vExportedImages.size() << " keyframes ("
         << nExportedMapKFs << " map keyframes, "
         << nExportedLocalizationKFs << " localization keyframes) and "
         << mMPToPoint3DId.size() << " map points." << endl;
    cout << "COLMAP text model saved at: " << sparseDir << endl;
    cout << "Keyframe images saved at: " << imageDir << endl;
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
