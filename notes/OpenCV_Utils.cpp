#include "wizard/OpenCV_Utils.hpp"

namespace Wizard {
namespace Utils {
namespace OpenCV {

cv::Point pointwise_max(const cv::Point& a, const cv::Point& b) {
    return cv::Point(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y);
}
cv::Point pointwise_min(const cv::Point& a, const cv::Point& b) {
    return cv::Point(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y);
}

    
void draw_onto(const cv::Mat& canvas, const cv::Mat& stamp, const cv::Point& loc) {
    // Crop the stamp as necessary

    cv::Point stamp_size(stamp.cols, stamp.rows);
    cv::Point canvas_size(canvas.cols, canvas.rows);

    cv::Point stamp_upper_left(0, 0);
    cv::Point stamp_lower_right(stamp_size);

    stamp_upper_left = pointwise_max(stamp_upper_left, -loc);
    stamp_lower_right = pointwise_min(stamp_lower_right, canvas_size - loc);

    cv::Point footprint_size = stamp_lower_right - stamp_upper_left;

    if (footprint_size.x <= 0 || footprint_size.y <= 0) {
        return;
    }

    cv::Point canvas_upper_left = pointwise_max(cv::Point(0, 0), loc);
    cv::Point canvas_lower_right = canvas_upper_left + footprint_size;

    cv::Rect stamp_crop(stamp_upper_left, stamp_lower_right);
    cv::Rect stamp_footprint(canvas_upper_left, canvas_lower_right);

    //std::cout << stamp_upper_left.x << ", " << stamp_upper_left.y << "\t" << stamp_lower_right.x << ", " << stamp_lower_right.y << std::endl;
    //std::cout << canvas_upper_left.x << ", " << canvas_upper_left.y << "\t\t" << canvas_lower_right.x << ", " << canvas_lower_right.y << std::endl;

    stamp(stamp_crop).copyTo(canvas(stamp_footprint));
}

void draw_trapezoid_onto(const cv::Mat& canvas, const cv::Mat& stamp, Eigen::Vector2i top_left, Eigen::Vector2i bottom_left, double min_bound, double max_bound) {
    bool invert_y_when_sampling;
    if (top_left[1] == bottom_left[1]) {
        return;
    }
    else if (top_left[1] < bottom_left[1]) {
        invert_y_when_sampling = false;
    }
    else {
        std::swap(top_left, bottom_left);
        invert_y_when_sampling = true;
    }


    for (int32_t delta_y = 0; delta_y < bottom_left[1] - top_left[1]; ++delta_y) {
        double interp_amount = delta_y;
        interp_amount /= bottom_left[1] - top_left[1];

        // Location within canvas
        int32_t row_y = top_left[1] + delta_y;
        int32_t row_x;
        {
            double interp = (top_left[0] * (1.0 - interp_amount)) + (bottom_left[0] * interp_amount);
            row_x = interp;
        }

        int32_t sample_y;
        {
            double fraction_of_the_way_down_the_orignal_strip = interp_amount;
            if (invert_y_when_sampling) {
                fraction_of_the_way_down_the_orignal_strip = 1.0 - fraction_of_the_way_down_the_orignal_strip;
            }

            fraction_of_the_way_down_the_orignal_strip *= (max_bound - min_bound);
            fraction_of_the_way_down_the_orignal_strip += min_bound;

            sample_y = fraction_of_the_way_down_the_orignal_strip * stamp.rows;
        }

        if (invert_y_when_sampling) {
            sample_y = stamp.rows - 1 - sample_y;
        }

        if (sample_y < 0) sample_y = 0;
        if (sample_y >= stamp.rows) sample_y = stamp.rows - 1;

        for (int32_t sample_x = 0; sample_x < stamp.cols; ++sample_x) {
            // Location within canvas
            int32_t pixel_x = row_x + sample_x;
            int32_t pixel_y = row_y;

            // Render
            if (pixel_x >= 0 && pixel_x < canvas.cols && pixel_y >= 0 && pixel_y < canvas.rows) {
                canvas.data[pixel_x + pixel_y * canvas.cols] =
                    stamp.data[sample_x + sample_y * stamp.cols];
            }
        }
    }
}

void draw_targetting_diamond(cv::Mat& canvas, const cv::Scalar& color, Eigen::Vector2i loc, int32_t size, bool use_mask_color, const cv::Scalar& mask_color) {
    if (canvas.channels() == 3 && canvas.depth() == CV_8U) {
        /*
        int32_t min_y = loc[1] - radius;
        int32_t max_y = loc[1] + radius + 1;
        int32_t min_x = loc[0] - radius;
        int32_t max_x = loc[0] + radius + 1;

        min_y = clamp(min_y, 0, canvas.rows);
        max_y = clamp(max_y, 0, canvas.rows);
        min_x = clamp(min_x, 0, canvas.cols);
        max_x = clamp(max_x, 0, canvas.cols);
        */

        /*
        int32_t min_rad = size / 2;
        int32_t max_rad = min_rad;
        if (size % 2 == 1) {
            max_rad += 1;
        }
        */

        for (int32_t dy = -size; dy <= size; ++dy) {
            for (int32_t dx = -size; dx <= size; ++dx) {

                int32_t x = loc[0] + dx;
                int32_t y = loc[1] + dy;

                if (x < 0 || x >= canvas.cols || y < 0 || y >= canvas.rows) continue;

                if (std::abs(dx) + std::abs(dy) > size) continue;

                if (use_mask_color) {
                    auto& pixel = canvas.at<cv::Vec3b>(y, x);

                    bool overwrite = true;
                    for (int32_t c = 0; c < 3; ++c) {
                        if (pixel[c] != mask_color[c]) {
                            overwrite = false;
                        }
                    }

                    if (overwrite) {
                        for (int32_t c = 0; c < 3; ++c) {
                            pixel[c] = color[c];
                        }
                    }
                }
                else {
                    auto& pixel = canvas.at<cv::Vec3b>(y, x);
                    for (int32_t c = 0; c < 3; ++c) {
                        pixel[c] = color[c];
                    }
                }
            }
        }
    }
    else {
        // Bit depth not supported
        std::cout << "Not implemented error reached 299550971" << std::endl; throw std::runtime_error("Not implemented 299550971");
    }
}
    
static bool in_bounds(const cv::Point &p, uint32_t width, uint32_t height) {
    return p.x >= 0 && p.y >= 0 && p.x < width && p.y < height;
}
   
    
void draw_line_onto(cv::Mat& canvas, const cv::Point2f& start, const cv::Point2f& end, uchar R, uchar G, uchar B) {
    // Inefficient and not great line graphics, but fast impl for what is needed now...
    auto vec = end - start;
    float len = sqrtf(vec.x * vec.x + vec.y * vec.y);
    vec *= 1.0f / len;
    for (float t = 0; t < len; t += 1.f) {
        cv::Point p = start + t * vec;
        if (in_bounds(p, canvas.cols, canvas.rows)) {
            //canvas.at<float>(p) = value;  // doesn't work
            
            // JAMES: The canvas now is only a single channel...
            canvas.data[(p.y * canvas.cols + p.x) + 0] = B;
            /*
            canvas.data[3 * (p.y * canvas.cols + p.x) + 0] = B;
            canvas.data[3 * (p.y * canvas.cols + p.x) + 1] = G;
            canvas.data[3 * (p.y * canvas.cols + p.x) + 2] = R;
            */
        }
    }
}

    
void draw_crosshair_onto(cv::Mat& canvas, const cv::Point2f& loc, float crosshair_length, uchar R, uchar G, uchar B) {
    draw_line_onto(canvas,
                   cv::Point(loc.x - 0.5f * crosshair_length, loc.y),
                   cv::Point(loc.x + 0.5f * crosshair_length, loc.y),
                   R, G, B);
    draw_line_onto(canvas,
                   cv::Point(loc.x, loc.y - 0.5f * crosshair_length),
                   cv::Point(loc.x, loc.y + 0.5f * crosshair_length),
                   R, G, B);
}
    

void unoptimized_naive_cross_correlation(const cv::Mat& target_img, const cv::Mat& strip, cv::Point& output_loc, double& output_peak_size) {

    // Padding on the x and y directions
    double padding_x = strip.cols - 1;
    double padding_y = strip.rows - 1;

    // Total size of the padded target_img
    double padded_width = target_img.cols + (2 * padding_x);
    double padded_height = target_img.rows + (2 * padding_y);

    // The fill value should just be the average of the whole image
    cv::Scalar fill_value = cv::sum(target_img) / (target_img.cols * target_img.rows);

    // Create the padded image, initializing with the "fill_value"
    int padded_img_shape[] = { (int) padded_height, (int) padded_width };
    cv::Mat padded_img = cv::Mat(2, padded_img_shape, target_img.type(), fill_value);
    
    // Copy the original target_img into the padded_img
    cv::Rect roi(cv::Point(padding_x, padding_y), cv::Size(target_img.cols, target_img.rows));
    target_img.copyTo(padded_img(roi));
    
    // Do the matching
    cv::Mat result;
    result.create(padded_img.rows - strip.rows + 1, padded_img.cols - strip.cols + 1, CV_32FC1);
    cv::matchTemplate(padded_img, strip, result, cv::TM_CCOEFF_NORMED);

    // Find the minimum and max values
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    // Output
    output_loc = max_loc - cv::Point(padding_x, padding_y);
    output_peak_size = max_val;
}


Precomputed_Unoptimized_OpenCV_Cross_Correlation::Precomputed_Unoptimized_OpenCV_Cross_Correlation(const cv::Mat& search_space, cv::Size pattern_size) {
    // Padding on the x and y directions
    m_padding_x = pattern_size.width - 1;
    m_padding_y = pattern_size.height - 1;

    // Total size of the padded target_img
    double padded_width = search_space.cols + (2 * m_padding_x);
    double padded_height = search_space.rows + (2 * m_padding_y);

    // The fill value should just be the average of the whole image
    cv::Scalar fill_value = cv::sum(search_space) / (search_space.cols * search_space.rows);

    // Create the padded image, initializing with the "fill_value"
    int padded_img_shape[] = { (int) padded_height, (int) padded_width };
    m_padded_img = cv::Mat(2, padded_img_shape, search_space.type(), fill_value);

    // Copy the original target_img into the padded_img
    cv::Rect roi(cv::Point(m_padding_x, m_padding_y), cv::Size(search_space.cols, search_space.rows));
    search_space.copyTo(m_padded_img(roi));

    //cv::namedWindow("Precomputed_Unoptimized_OpenCV_Cross_Correlation");
}

void Precomputed_Unoptimized_OpenCV_Cross_Correlation::match(const cv::Mat& pattern, cv::Point& output_loc, double& output_peak_size) const {

    // Do the matching
    cv::Mat result;
    result.create(m_padded_img.rows - pattern.rows + 1, m_padded_img.cols - pattern.cols + 1, CV_32FC1);


    cv::matchTemplate(m_padded_img, pattern, result, cv::TM_CCOEFF_NORMED);

    // Find the minimum and max values
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    // Output
    output_loc = max_loc - cv::Point(m_padding_x, m_padding_y);
    output_peak_size = max_val;


}

void Precomputed_Unoptimized_OpenCV_Cross_Correlation::match_extra(const cv::Mat& pattern, cv::Point& output_loc, double& output_peak_size, double& sigma_rating) const {

    // Original Python implementation:
    /*
    
    # Register a strip onto target_img
    def cross_correlate(target_img, strip, strategy=cv2.TM_CCOEFF_NORMED):
  
      padding_x = strip.shape[0] - 1
      padding_y = strip.shape[1] - 1
  
      padded_width = target_img.shape[0] + 2 * padding_x
      padded_height = target_img.shape[1] + 2 * padding_y
  
      fill_value = np.mean(target_img)
  
      padded_img = np.full((padded_width, padded_height), fill_value, target_img.dtype)
      padded_img[padding_x:padding_x+target_img.shape[0],padding_y:padding_y+target_img.shape[1]] = target_img
  
      cross = cv2.matchTemplate(padded_img, strip, strategy)
      _, _, _, max_loc = cv2.minMaxLoc(cross)
  
      padding = np.array((padding_x, padding_y))
      #pos = max_loc - padding
      pos = np.array(max_loc) - np.array((padding[1], padding[0]))
  
      cross_mean = np.mean(cross)
      cross_sigma = np.std(cross)
      sigma_rating = np.abs(cross_mean - np.max(cross)) / cross_sigma
    
      return pos, sigma_rating, cross, padded_img
    */

    // Do the matching
    cv::Mat cross;
    cross.create(m_padded_img.rows - pattern.rows + 1, m_padded_img.cols - pattern.cols + 1, CV_32FC1);

    cv::matchTemplate(m_padded_img, pattern, cross, cv::TM_CCOEFF_NORMED);

    // Find the minimum and max values
    double peak_value = 0;
    Eigen::Vector2i peak_loc;
    double mean, std_dev;
    {
        int32_t num_cols = cross.cols;
        int32_t num_rows = cross.rows;
        mean = 0;
        std_dev = 0;
        for (int32_t r = 0; r < num_rows; ++r) {
            for (int32_t c = 0; c < num_cols; ++c) {
                double val = cross.at<float>(r, c);
                mean += val;
                if (val >= peak_value) {
                    peak_value = val;
                    peak_loc[0] = c;
                    peak_loc[1] = r;
                }
            }
        }
        mean /= num_rows * num_cols;
        
        for (int32_t r = 0; r < num_rows; ++r) {
            for (int32_t c = 0; c < num_cols; ++c) {
                double val = cross.at<float>(r, c);
                double diff = val - mean;
                std_dev += diff*diff;
            }
        }
        std_dev /= num_rows * num_cols;
        std_dev = std::sqrt(std_dev);
    }
    // Output
    output_loc = cv::Point(peak_loc[0], peak_loc[1]) - cv::Point(m_padding_x, m_padding_y);
    output_peak_size = peak_value;
    sigma_rating = std::abs(output_peak_size - mean) / std_dev;



	//split it in half
	cv::Mat pattern1 = cv::Mat(16, 256, CV_8U);
	cv::Mat	pattern2 = cv::Mat(16, 256, CV_8U);

	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 16; j++) {
			pattern1.at<uchar>(j, i) = pattern.at<uchar>(j, i);
			pattern2.at<uchar>(j, i) = pattern.at<uchar>(j, i + 256);
		}
	}

	double minVal;
	double maxVal;
	cv::Point minLoc1;
	cv::Point maxLoc1;
	cv::Point minLoc2;
	cv::Point maxLoc2;

	cv::Mat cross1, cross2;
	cross1.create(m_padded_img.rows - pattern1.rows + 1, m_padded_img.cols - pattern1.cols + 1, CV_32FC1);
	cv::matchTemplate(m_padded_img, pattern1, cross1, cv::TM_CCOEFF_NORMED);
	cv::minMaxLoc(cross1, &minVal, &maxVal, &minLoc1, &maxLoc1);

	cross2.create(m_padded_img.rows - pattern2.rows + 1, m_padded_img.cols - pattern2.cols + 1, CV_32FC1);
	cv::matchTemplate(m_padded_img, pattern2, cross2, cv::TM_CCOEFF_NORMED);
	cv::minMaxLoc(cross2, &minVal, &maxVal, &minLoc2, &maxLoc2);

	std::cout << "left" << maxLoc1 << std::endl;
	std::cout << "right" << maxLoc2 << std::endl;

    //cv::circle(cross, cv::Point(peak_loc[0], peak_loc[1]), 3, cv::Scalar(255, 255, 255));
    //cv::imshow("Precomputed_Unoptimized_OpenCV_Cross_Correlation", cross);
    
    /*
    std::cout << "====================" << std::endl;
    std::cout << "cross_mean: " << mean << std::endl;
    std::cout << "cross_sigma: " << std_dev << std::endl;
    std::cout << "peak_value: " << peak_value << std::endl;
    std::cout << "output_loc: " << output_loc << std::endl;
    std::cout << "sigma_rating: " << sigma_rating << std::endl;
    std::cout << "====================" << std::endl;
    /**/
    //std::cout << "sigma_rating: " << sigma_rating << std::endl;
}

cv::Scalar cone_type_to_color(Core::Bio_Cone_Type cone_type, float max_intensity) {
    switch (cone_type) {
        case Core::Bio_Cone_Type::SHORT_WAVELENGTH: return cv::Scalar(0, 0, max_intensity);
        case Core::Bio_Cone_Type::MEDIUM_WAVELENGTH: return cv::Scalar(0, max_intensity, 0);
        case Core::Bio_Cone_Type::LONG_WAVELENGTH: return cv::Scalar(max_intensity, 0, 0);
        default: return cv::Scalar(0, 0, 0);
    }
}

cv::Mat remove_all_but_first_channel(const cv::Mat& original_image) {
    // Make a new pre-allocated image of "original_image"'s color depth, but with only 1 channel
    cv::Mat retval(original_image.size(), CV_MAKE_TYPE(original_image.depth(), 1));

    // Copy channel 0 of the input to channel 0 of the output
    int from_to[] = { 0, 0 };

    cv::mixChannels(&original_image, 1, &retval, 1, from_to, 1);

    return retval;
}

void draw_text(const cv::Mat& canvas, std::string text, Eigen::Vector2i loc, double font_scale) {
    double font_base_size = 25;
    cv::putText(canvas, text, cv::Point(loc[0], loc[1]), cv::FONT_HERSHEY_DUPLEX, font_scale / font_base_size, cv::Scalar(0), 3);
    cv::putText(canvas, text, cv::Point(loc[0], loc[1]), cv::FONT_HERSHEY_DUPLEX, font_scale / font_base_size, cv::Scalar(255), 1.5);
}

} // namespace OpenCV
} // namespace Default
} // namespace Wizard
