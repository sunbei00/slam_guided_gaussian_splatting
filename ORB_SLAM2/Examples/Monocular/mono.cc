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


#include<iostream>
#include <unistd.h>
#include<algorithm>
#include<cctype>
#include<dirent.h>
#include<chrono>
#include<sys/stat.h>
#include<utility>
#include<vector>

#include<opencv2/core/core.hpp>
#if CV_MAJOR_VERSION >= 3
#include<opencv2/core/ocl.hpp>
#endif

#include"System.h"

using namespace std;

void LoadImages(const string &strImagesPath, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);
int LoadMaxImages(const string &strSettingsPath);

int main(int argc, char **argv)
{
    if(argc != 4 && argc != 5)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_images [output_colmap_dir]" << endl;
        return 1;
    }

    cv::setNumThreads(1);
    cv::setRNGSeed(0);
#if CV_MAJOR_VERSION >= 3
    cv::ocl::setUseOpenCL(false);
#endif

    const string strOutputColmapDir = (argc==5) ? string(argv[4]) : string("colmap_orbslam2_export");
    const int nMaxImages = LoadMaxImages(string(argv[2]));

    // Retrieve sorted image paths and synthesize timestamps at 10 Hz.
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);

    if(nMaxImages > 0 && static_cast<size_t>(nMaxImages) < vstrImageFilenames.size())
    {
        vstrImageFilenames.resize(nMaxImages);
        vTimestamps.resize(nMaxImages);
    }

    int nImages = vstrImageFilenames.size();
    if(nImages == 0)
    {
        cerr << endl << "No images found in: " << argv[3] << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    // Export final map as COLMAP text model using SLAM keyframes and map points
    SLAM.SaveMapToCOLMAP(strOutputColmapDir, vstrImageFilenames, argv[2]);

    return 0;
}

namespace
{

string JoinPath(const string &dir, const string &name)
{
    if(dir.empty())
        return name;
    if(dir[dir.size()-1] == '/')
        return dir + name;
    return dir + "/" + name;
}

string ToLower(string value)
{
    for(size_t i=0; i<value.size(); ++i)
        value[i] = static_cast<char>(tolower(static_cast<unsigned char>(value[i])));
    return value;
}

bool HasImageExtension(const string &filename)
{
    const size_t pos = filename.find_last_of('.');
    if(pos == string::npos)
        return false;

    const string ext = ToLower(filename.substr(pos + 1));
    return ext == "png" || ext == "jpg" || ext == "jpeg" ||
           ext == "bmp" || ext == "tif" || ext == "tiff";
}

bool IsRegularFile(const string &path)
{
    struct stat st;
    if(stat(path.c_str(), &st) != 0)
        return false;
    return S_ISREG(st.st_mode);
}

} // namespace

int LoadMaxImages(const string &strSettingsPath)
{
    cv::FileStorage fsSettings(strSettingsPath, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "Failed to open settings file: " << strSettingsPath << endl;
        return 0;
    }

    const cv::FileNode node = fsSettings["Input.MaxImages"];
    if(node.empty())
        return 0;

    return static_cast<int>(node);
}

void LoadImages(const string &strImagesPath, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    DIR *dir = opendir(strImagesPath.c_str());
    if(!dir)
    {
        cerr << endl << "Failed to open image directory: " << strImagesPath << endl;
        return;
    }

    vector<pair<string, string> > vImageEntries;
    struct dirent *entry;
    while((entry = readdir(dir)) != NULL)
    {
        const string filename = entry->d_name;
        if(filename == "." || filename == ".." || !HasImageExtension(filename))
            continue;

        const string imagePath = JoinPath(strImagesPath, filename);
        if(IsRegularFile(imagePath))
            vImageEntries.push_back(make_pair(filename, imagePath));
    }
    closedir(dir);

    sort(vImageEntries.begin(), vImageEntries.end());

    vstrImageFilenames.clear();
    vTimestamps.clear();
    vstrImageFilenames.reserve(vImageEntries.size());
    vTimestamps.reserve(vImageEntries.size());

    for(size_t i=0; i<vImageEntries.size(); ++i)
    {
        vstrImageFilenames.push_back(vImageEntries[i].second);
        vTimestamps.push_back(0.1 * static_cast<double>(i));
    }
}
