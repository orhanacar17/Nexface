# 👁️ NexFace

A powerful, local, and user-friendly desktop application for high-quality batch face swapping. Built with Python and Gradio, this tool provides a convenient interface for the Face2Face library, allowing you to swap a single source face onto hundreds of target images or videos with just a few clicks.

 
*(Note: You should replace this with a real screenshot or GIF of your application in action.)*

## ✨ Key Features

-   **Batch Processing**: Swap a face onto an entire folder of target images or videos in one go.
-   **Video Processing**: Support for both single video and batch video processing with audio preservation.
-   **High-Quality Mode**: An optional mode that intelligently preprocesses images, upscales faces using Lanczos resampling, and applies sharpening and contrast enhancements for a more professional result.
-   **Face Enhancement**: Supports multiple face enhancement models including GFPGAN and GPEN for superior quality results.
-   **User-Friendly Interface**: A clean and simple UI powered by Gradio with three main tabs:
  - Image Processing
  - Video Processing
  - Batch Video Processing
-   **Convenient Output Management**: Automatically saves all results into an `output` folder and provides a single `.zip` file for easy download.
-   **Local & Private**: Runs entirely on your own machine. Your images are never uploaded to any cloud service.
-   **Cross-Platform**: Works on Windows, macOS, and Linux. Includes a handy button to open the output folder in your native file explorer.
-   **Progress Tracking**: Real-time progress updates with estimated time remaining for both image and video processing.
-   **Memory Management**: Automatic GPU memory cleanup and garbage collection to prevent out-of-memory errors.


## ⚙️ How It Works

This application is built on top of the Face2Face library, which uses Insightface's powerful face detection and swapping technology. When "High Quality Mode" is enabled, the script performs several additional steps:

1.  **Preprocessing**: Source and target images are upscaled to a minimum size (1024px) using cubic interpolation to provide more detail for the swap.
2.  **Swapping**: The Face2Face library performs the core face swapping using Insightface's models.
3.  **Post-processing**: The swapped face region is enhanced using a combination of high-quality upscaling (Pillow's `LANCZOS` filter), sharpening, and a subtle contrast boost.
4.  **Face Enhancement**: Optional face enhancement using models like GFPGAN or GPEN can be applied for superior quality results.

![Image](https://github.com/user-attachments/assets/ba87ae3f-84ba-4ec0-8490-da22f9f1efdf)

## 📦 Requirements

-   Python 3.7 or higher
-   Minimum 5GB of RAM
-   Recommended: GPU with at least 8GB of VRAM for fast inference (runs on CPU but will be slower)
-   For video support: FFmpeg installation
-   The Python libraries listed in `requirements.txt`.

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ExoFi-Labs/Nexface.git
    cd nexface
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    # For basic face swapping functionality
    pip install socaity-face2face
    
    # For full functionality including web service
    pip install socaity-face2face[full]
    
    # Install additional project requirements
    pip install -r requirements.txt
    ```

## ▶️ How to Run

1.  Navigate to the project directory in your terminal.
2.  Run the application script:
    ```bash
    python app.py 
    ```
    
3.  The script will start a local web server and print a URL (usually `http://127.0.0.1:7860`). Open this URL in your web browser to use the application.

## 📖 How to Use

### Image Processing
1. **Upload Source Face**: Drag and drop an image containing the face you want to use into the "Source Face" box.
2. **Upload Target Images**: Drag and drop one or more images you want to swap the face onto into the "Target Images" box.
3. **Configure Options**:
    - Check **✨ Enhance Output Faces** for an extra layer of clarity.
    - Check **🔍 High Quality Mode** for the best possible results (this will be slower).
    - Adjust the **Face Upscale Factor** if using High Quality Mode.
4. **Start Processing**: Click the **🔁 Start Processing** button.
5. **View Results**:
    - Previews of the swapped images will appear in the "Swapped Results" gallery.
    - Once the batch is complete, a link to download a `.zip` file of all results will appear.
    - Click the **📂 Open Output Directory** button to directly access the saved image files on your computer.

### Video Processing
1. **Upload Source Face**: Drag and drop an image containing the face you want to use.
2. **Upload Target Video**: Select a video file you want to process.
3. **Configure Options**: Same as image processing.
4. **Start Processing**: Click the **🔁 Start Video Processing** button.
5. **View Results**: The processed video will appear in the output section with audio preserved.

### Batch Video Processing
1. **Upload Source Face**: Drag and drop an image containing the face you want to use.
2. **Upload Target Videos**: Select multiple video files to process.
3. **Configure Options**: Same as image processing.
4. **Start Processing**: Click the **🔁 Start Batch Processing** button.
5. **View Results**: A `.zip` file containing all processed videos will be available for download.

## ⚠️ Known Limitations

- Processing large videos may require significant GPU memory. The application includes automatic memory management, but very large videos might still cause issues on systems with limited VRAM.
- Video processing speed depends on your hardware. GPU acceleration is recommended for faster processing.
- The application requires FFmpeg for video processing with audio. If FFmpeg is not installed, videos will be processed without audio.

## ⚠️ Ethical Disclaimer

This tool is intended for creative, educational, and ethical purposes only. The creation of non-consensual content, deepfakes, or any material that could be used to mislead, harass, or defame individuals is strictly prohibited. By using this software, you agree to use it responsibly and in accordance with all applicable laws. The developers of this tool are not responsible for any misuse.

## 🙏 Acknowledgments

This project is built on top of several amazing open-source projects:

- [Insightface](https://github.com/deepinsight/insightface) - For the core face detection and swapping technology
- [ROOP](https://github.com/s0md3v/roop) - For inspiration and parts of the face swapping implementation
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - For face restoration capabilities
- [GPEN](https://github.com/yangxy/GPEN) - For additional face enhancement options

A special thanks to all the contributors of these projects for making this technology accessible to the community.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
