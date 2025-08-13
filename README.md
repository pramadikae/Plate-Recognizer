# Plate-Recognizer

This project provides a comprehensive solution for License Plate Recognition (LPR), leveraging YOLOv5 for efficient license plate detection and a custom Convolutional Neural Network (CNN) for accurate character recognition. The application is built with Streamlit, offering an intuitive web interface for training, detection, and recognition tasks.

## Features

-   **License Plate Detection**: Utilizes YOLOv5 to accurately detect license plates in images.
-   **Character Recognition**: Employs a custom-trained CNN model to recognize characters on detected license plates.
-   **Streamlit Web Interface**: A user-friendly web application for:
    -   **Training**: Upload datasets and train custom YOLOv5 models.
    -   **Detection**: Upload images and perform license plate detection.
    -   **Recognition**: Process detected license plates to recognize characters.
-   **Modular Codebase**: Organized into logical modules for better maintainability and scalability.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

-   Python 3.8+
-   `pip` (Python package installer)
-   Git

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/Plate-Recognizer.git
    cd Plate-Recognizer
    ```

    *(Note: Replace `your-username` with the actual GitHub username if this project is hosted on GitHub.)*

2.  **Create a virtual environment (recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Clone YOLOv5 (if not already present)**:
    This project relies on the YOLOv5 repository. Clone it into the root of this project:

    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```

### Running the Application

To start the Streamlit application, run the following command from the project root directory:

```bash
streamlit run src/app/home.py
```

This will open the application in your default web browser.

## Project Structure

The project is organized as follows:

```
.
├── data/                     # Directory for raw and processed datasets
├── models/                   # Stores trained YOLOv5 and character recognition models
├── notebooks/                # Jupyter notebooks for experimentation and development
│   ├── character recognition.ipynb
│   └── plate detection.ipynb
├── src/                      # Main source code
│   ├── app/                  # Streamlit application files
│   │   ├── home.py           # Main Streamlit entry point
│   │   └── pages/            # Individual Streamlit pages
│   │       ├── 1_🔁_Training.py
│   │       ├── 2_📸_Detection.py
│   │       └── 3_Recognizer.py
│   ├── character_recognition/# Modules for character recognition logic
│   │   └── recognizer.py
│   ├── plate_detection/      # Modules for license plate detection logic
│   │   └── detector.py
│   └── utils/                # Utility functions and helper scripts
│       └── file_utils.py
├── .gitignore                # Specifies intentionally untracked files to ignore
├── README.md                 # Project overview and setup instructions
└── requirements.txt          # Python dependencies
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.