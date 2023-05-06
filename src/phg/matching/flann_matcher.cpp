#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat ind, dst;

    flann_index->knnSearch(query_desc, ind, dst, k, *search_params);

    for (int i = 0; i < query_desc.rows; i++) {
        matches.push_back(std::vector<cv::DMatch>());
        for (int j = 0; j < k; j++) {
            matches.back().push_back(cv::DMatch(i, ind.at<int>(i, j), dst.at<float>(i, j)));
        }
    }
}
