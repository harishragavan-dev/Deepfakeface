# Deepfake Face Swap Project

## Description

This project involves processing video frames to detect faces and swap them with faces from source images. The script can handle multiple target images and perform face swaps if matches are found.

## Requirements

1. **Python 3.x**: Ensure Python 3.x is installed on your system.
2. **Dependencies**: Install the required Python packages using the provided `requirements.txt` file.

## Installation

1. **Clone the Repository**:

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Set Up a Virtual Environment (Optional but recommended)**:

    ```sh
    python -m venv env
    ```

3. **Activate the Virtual Environment**:

    - On Windows:

      ```sh
      .\env\Scripts\activate
      ```

    - On macOS/Linux:

      ```sh
      source env/bin/activate
      ```

4. **Install the Dependencies**:

    ```sh
    pip install -r requirements.txt
    ```
4. **Install the Dependencies**:

    ```sh
   place the "inswapper_128.onnx" in models folder
    ```

## Usage

Run the script using the following command:

```sh
python main.py -tv <video_path> -s <source_image_path> -o <output_video_path> [-t <target_image_path>] [-tf <target_file_path>]
