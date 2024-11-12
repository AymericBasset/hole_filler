import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io


def main():
    st.title("Image Thresholding and Noise Reduction App")
    st.write(
        "Upload an image and use the sliders to adjust the dilation and erosion parameters.")

    # Upload image
    uploaded_file = st.file_uploader("Choose a JPG image", type="jpg")
    if uploaded_file is not None:
        # Convert uploaded file to a usable format for OpenCV
        image = Image.open(uploaded_file).convert('RGB')
        image = np.array(image)

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        col1, col2 = st.columns(2)
        with col1:
            # Sliders to adjust morphological operations
            dilation_kernel_size = st.slider(
                "Dilation kernel size", min_value=1, max_value=15, value=25, step=1)
            dilation_iterations = st.slider(
                "Dilation iterations", min_value=1, max_value=20, value=1, step=1)
            erosion_kernel_size = st.slider(
                "Erosion kernel size", min_value=1, max_value=25, value=5, step=2)
            erosion_iterations = st.slider(
                "Erosion iterations", min_value=1, max_value=20, value=1, step=1)
            hole_size = st.slider("Max hole size to fill",
                                  min_value=1, max_value=10000, value=50, step=10)
            top_n_components = st.slider("Number of largest connected components to keep",
                                         min_value=1, max_value=10, value=1, step=1)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold the image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Create kernels for morphological operations
        dilation_kernel = np.ones(
            (dilation_kernel_size, dilation_kernel_size), np.uint8)
        erosion_kernel = np.ones(
            (erosion_kernel_size, erosion_kernel_size), np.uint8)

        # Perform morphological operations to remove noise
        dilation = cv2.dilate(binary, dilation_kernel,
                              iterations=dilation_iterations)
        erosion = cv2.erode(dilation, erosion_kernel,
                            iterations=erosion_iterations)
        cleaned = erosion

        # Fill small holes
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < hole_size:
                cv2.drawContours(cleaned, [contour], 0, 255, -1)

        # Keep the top n largest connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8)
        sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
        top_components = sorted_indices[:top_n_components]
        maximal_area = np.zeros_like(cleaned)
        for component in top_components:
            maximal_area[labels == component] = 255

        # Calculate area of the ones and the ratio of ones to zeros
        ones_area = np.sum(maximal_area == 255)
        total_area = maximal_area.size
        ratio_ones_to_zeros = ones_area / (total_area - ones_area)

        with col2:
            # Display the processed image
            st.image(maximal_area, caption='Processed Image with Top Connected Areas',
                     use_column_width=True)
        st.write(f"Area of ones: {ones_area} pixels")
        st.write(f"Ratio of ones to zeros: {ratio_ones_to_zeros:.4f}")

        # Add an export button to download the final image
        result_image = Image.fromarray(maximal_area)
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Processed Image", data=byte_im,
                           file_name="processed_image.png", mime="image/png")

    # Step to upload a new image to apply mask
    st.write("Upload a new image to apply the mask.")
    new_uploaded_file = st.file_uploader(
        "Choose another JPG image", type="jpg", key="new_image")
    if new_uploaded_file is not None:
        new_image = Image.open(new_uploaded_file).convert('RGB')
        st.image(new_image, caption='astrocytes',
                 use_column_width=True)
        new_image = np.array(new_image)

        # Convert new image to grayscale
        new_gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)

        # Slider to adjust threshold for new image
        new_threshold_value = st.slider(
            "Threshold value for new image", min_value=0, max_value=255, value=127, step=1, key="new_threshold")

        # Apply threshold to binarize the new image
        _, new_binary = cv2.threshold(
            new_gray, new_threshold_value, 255, cv2.THRESH_BINARY)

        st.image(new_binary, caption='thresholded astrocytes',
                 use_column_width=True)
        # Calculate the proportion of ones in the mask area
        mask_area = maximal_area == 255
        ones_in_mask = np.sum(new_binary[mask_area] == 255)
        total_mask_area = np.sum(mask_area)
        proportion_ones = ones_in_mask / total_mask_area
        proportion_ones_2 = ones_in_mask / ones_area

        # Display the result
        st.write(
            f"Proportion of pixels that are ones in the mask area ?: {proportion_ones:.4f}")
        st.write(
            f"Proportion of pixels that are ones in the mask area: {proportion_ones_2:.4f}")

        # Add an export button to download the thresholded new image
        new_threshold_image = Image.fromarray(new_binary)
        buf_new_threshold = io.BytesIO()
        new_threshold_image.save(buf_new_threshold, format="PNG")
        byte_new_threshold = buf_new_threshold.getvalue()
        st.download_button(label="Download Thresholded New Image", data=byte_new_threshold,
                           file_name="new_thresholded_image.png", mime="image/png")


if __name__ == "__main__":
    main()
