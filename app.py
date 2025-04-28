import streamlit as st

# Set page config as the first Streamlit command
st.set_page_config(page_title="First Order Motion Model Demo", layout="wide")

# Now import other libraries
import PIL.Image
import cv2
import numpy as np
import os
import os.path
import requests
import skimage.transform
import warnings
import tempfile
import subprocess
from shutil import copyfileobj
from skimage import img_as_ubyte
from tqdm.auto import tqdm

# Install required packages if missing
missing_packages = []

# First, make sure we have the correct ffmpeg module
try:
    import ffmpeg
except ModuleNotFoundError:
    missing_packages.append("ffmpeg-python")

# Also try to import imageio with ffmpeg capabilities
try:
    import imageio
    import imageio_ffmpeg
except ModuleNotFoundError:
    missing_packages.append("imageio")
    missing_packages.append("imageio-ffmpeg")

# Install missing packages if any
if missing_packages:
    st.warning(f"Installing required packages: {', '.join(missing_packages)}")
    for package in missing_packages:
        subprocess.check_call(["pip", "install", package])
    
    # Import after installation
    import ffmpeg
    import imageio
    import imageio_ffmpeg
    st.success("Packages installed successfully!")

# Configuration
warnings.filterwarnings("ignore")
os.makedirs("user", exist_ok=True)

# Custom CSS
st.markdown("""
<style>
.main {
    max-width: 1200px;
}
.resource {
    cursor: pointer;
    border: 1px solid gray;
    margin: 5px;
    width: 160px;
    height: 160px;
}
.resource:hover {
    border: 6px solid crimson;
    margin: 0;
}
.selected {
    border: 6px solid seagreen;
    margin: 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    font-size: 15px;
}
h1 {
    font-size: 24px;
    font-weight: bold;
    margin: 12px 0 6px 0;
}
h2 {
    font-size: 20px;
    font-weight: bold;
    margin: 12px 0 6px 0;
}
.preview-container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 266px;
    height: 266px;
    border: 1px solid gray;
    background-color: #f0f0f0;
}
.uploaded {
    border: 6px solid seagreen;
    margin: 0;
}
.output-container {
    display: flex;
    justify-content: space-between;
}
.video-container {
    width: 256px;
    height: 256px;
    border: 1px solid gray;
}
.comparison-label {
    color: gray;
    font-size: 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Setup repositories if they don't exist
if not os.path.isdir("first-order-model"):
    st.info("Setting up necessary repositories...")
    try:
        subprocess.run(["git", "clone", "https://github.com/AliaksandrSiarohin/first-order-model"], check=True)
        if not os.path.isdir("demo"):
            subprocess.run(["git", "clone", "https://github.com/graphemecluster/first-order-model-demo", "demo"], check=True)
        st.success("Repositories successfully cloned!")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to clone repositories: {e}")
        st.stop()

# Helper functions
def thumbnail(file):
    try:
        return imageio.get_reader(file, mode='I', format='FFMPEG').get_next_data()
    except Exception as e:
        st.warning(f"Error loading video thumbnail: {e}")
        # Return a blank image as fallback
        return np.zeros((256, 256, 3), dtype=np.uint8)

def resize(image, size=(256, 256)):
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    
    w, h = image.size
    d = min(w, h)
    r = ((w - d) // 2, (h - d) // 2, (w + d) // 2, (h + d) // 2)
    return image.resize(size, resample=PIL.Image.LANCZOS, box=r)

# Import the demo modules
@st.cache_resource
def load_modules():
    try:
        import sys
        import torch
        # Add the repositories to the Python path
        if "first-order-model" not in sys.path:
            sys.path.append("first-order-model")
        if "demo" not in sys.path:
            sys.path.append("demo")
        
        try:
            # Create a custom load_checkpoints function that avoids CUDA errors
            import yaml
            from modules.generator import OcclusionAwareGenerator
            from modules.keypoint_detector import KPDetector
            
            def load_checkpoints(config_path, checkpoint_path):
                with open(config_path) as f:
                    config = yaml.load(f, Loader=yaml.SafeLoader)
                
                generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                   **config['model_params']['common_params'])
                generator.eval()
                
                kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                        **config['model_params']['common_params'])
                kp_detector.eval()
                
                # Use CPU by default to avoid CUDA errors
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                generator.load_state_dict(checkpoint['generator'])
                kp_detector.load_state_dict(checkpoint['kp_detector'])
                
                # Only use CUDA if it's available and working
                if torch.cuda.is_available():
                    try:
                        # Test if CUDA is really working
                        test_tensor = torch.zeros(1).cuda()
                        # If we get here, CUDA is working
                        generator = generator.cuda()
                        kp_detector = kp_detector.cuda()
                    except:
                        # If there's any issue with CUDA, stay on CPU
                        st.warning("CUDA error detected. Using CPU instead.")
                else:
                    st.info("CUDA not available. Using CPU for processing (this will be slower).")
                
                generator.eval()
                kp_detector.eval()
                
                return generator, kp_detector
            
            # Also import the make_animation function
            from demo import make_animation
            
            return load_checkpoints, make_animation
            
        except ImportError as e:
            st.error(f"Could not import from demo: {e}")
            # Create a fallback implementation
            def load_checkpoints(*args, **kwargs):
                return None, None
            
            def make_animation(*args, **kwargs):
                return []
                
            return load_checkpoints, make_animation
    except Exception as e:
        st.error(f"Error loading modules: {e}")
        # Create fallback implementations
        def load_checkpoints(*args, **kwargs):
            return None, None
        
        def make_animation(*args, **kwargs):
            return []
            
        return load_checkpoints, make_animation

# Check if demo files exist
def check_demo_files():
    if not os.path.isdir("demo/images") or not os.path.isdir("demo/videos"):
        st.error("Demo image and video directories not found. Please make sure to clone the demo repository.")
        return False
    
    # Check at least one example file
    if not os.path.isfile("demo/images/00.png"):
        st.error("Demo images not found. Please make sure to clone the demo repository correctly.")
        return False
    
    if not os.path.isfile("demo/videos/0.mp4"):
        st.error("Demo videos not found. Please make sure to clone the demo repository correctly.")
        return False
    
    return True

# Initialize session state
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
    st.session_state.selected_image_path = "demo/images/00.png"
    st.session_state.uploaded_image = None
    
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
    st.session_state.selected_video_path = "demo/videos/0.mp4"
    st.session_state.uploaded_video = None

if "output_video" not in st.session_state:
    st.session_state.output_video = None
    st.session_state.comparison_video = None

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Run file check - only warn but continue
demo_files_exist = check_demo_files()

# Define image and video data
image_titles = ['Peoples', 'Cartoons', 'Dolls', 'Game of Thrones', 'Statues']
image_lengths = [8, 4, 8, 9, 4]

# Create config directory if it doesn't exist
if not os.path.isdir("config"):
    os.makedirs("config", exist_ok=True)

# Create config yaml files if they don't exist
config_models = ['vox', 'vox-adv', 'taichi', 'taichi-adv', 'nemo', 'mgif', 'fashion', 'bair']
for model_name in config_models:
    config_path = f"config/{model_name}-256.yaml"
    if not os.path.isfile(config_path):
        # Create a basic config file - this is a simplified example for vox model
        config_content = """
dataset_params:
  root_dir: data/vox-cpu
  frame_shape: [256, 256, 3]
  id_sampling: True
  augmentation_params:
    flip_param:
      horizontal_flip: True
      time_flip: True
    jitter_param:
      brightness: 0.1
      contrast: 0.1
      saturation: 0.1
      hue: 0.1

model_params:
  common_params:
    num_kp: 10
    num_channels: 3
    estimate_jacobian: True
  kp_detector_params:
    temperature: 0.1
    block_expansion: 32
    max_features: 1024
    scale_factor: 0.25
    num_blocks: 5
  generator_params:
    block_expansion: 64
    max_features: 512
    num_down_blocks: 2
    num_bottleneck_blocks: 6
    estimate_occlusion_map: True
    dense_motion_params:
      block_expansion: 64
      max_features: 1024
      num_blocks: 5
      scale_factor: 0.25

train_params:
  num_epochs: 100
  num_repeats: 75
  epoch_milestones: [60, 90]
  lr: 2.0e-4
  batch_size: 8
  scales: [1, 0.5, 0.25, 0.125]
  checkpoint_freq: 50
  transform_params:
    sigma_affine: 0.05
    sigma_tps: 0.005
    points_tps: 5
  loss_weights:
    perceptual: [10, 10, 10, 10, 10]
    equivariance_value: 10
    equivariance_jacobian: 10

reconstruction_params:
  num_videos: 1000
  format: '.mp4'

visualizer_params:
  kp_size: 5
  draw_border: True
  colormap: 'gist_rainbow'
"""
        with open(config_path, "w") as f:
            f.write(config_content)

# Main App UI
st.title("First Order Model Animation Demo")

# Main interface with tabs
tab1, tab2, tab3 = st.tabs(["Input", "Settings", "Output"])

with tab1:
    # Split the tab into two columns for image and video
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Choose Source Image")
        
        # Image upload section
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            try:
                image = PIL.Image.open(uploaded_image).convert("RGB")
                st.session_state.uploaded_image = resize(image)
                st.session_state.selected_image = st.session_state.uploaded_image
            except Exception as e:
                st.error(f"Error loading image: {e}")
        
        # Display the selected/uploaded image
        if st.session_state.selected_image is not None:
            st.image(st.session_state.selected_image, width=256)
        elif demo_files_exist:
            try:
                # Load the default image
                default_image = PIL.Image.open(st.session_state.selected_image_path).convert("RGB")
                st.session_state.selected_image = resize(default_image)
                st.image(st.session_state.selected_image, width=256)
            except Exception as e:
                st.error(f"Error loading default image: {e}")
        
        # Image tabs with examples (only show if demo files exist)
        if demo_files_exist:
            st.subheader("Or select from examples:")
            image_tab = st.tabs(image_titles)
            
            for i, tab in enumerate(image_tab):
                with tab:
                    # Create a grid of images
                    columns = st.columns(4)
                    for j in range(image_lengths[i]):
                        with columns[j % 4]:
                            img_path = f"demo/images/{i}{j}.png"
                            try:
                                if os.path.isfile(img_path):
                                    img = PIL.Image.open(img_path).convert("RGB")
                                    st.image(img, width=120, caption=f"Image {i}{j}")
                                    if st.button(f"Select {i}{j}", key=f"img_{i}{j}"):
                                        st.session_state.selected_image = resize(img)
                                        st.session_state.selected_image_path = img_path
                                        st.session_state.uploaded_image = None
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Error loading example image {i}{j}: {e}")
    
    with col2:
        st.header("Choose Driving Video")
        
        # Video upload section
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            try:
                # Save uploaded video to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_video.read())
                temp_file.close()
                
                st.session_state.uploaded_video = temp_file.name
                st.session_state.selected_video = st.session_state.uploaded_video
                
                # Display video preview
                video_preview = thumbnail(st.session_state.uploaded_video)
                st.image(video_preview, width=256, caption="Video Preview")
            except Exception as e:
                st.error(f"Error processing uploaded video: {e}")
        elif demo_files_exist:
            # Display default video preview
            try:
                video_preview = thumbnail(st.session_state.selected_video_path)
                st.image(video_preview, width=256, caption="Default Video")
            except Exception as e:
                st.error(f"Error loading default video: {e}")
        
        # Video selection (only show if demo files exist)
        if demo_files_exist:
            st.subheader("Or select from examples:")
            video_columns = st.columns(5)
            
            for i in range(5):
                with video_columns[i]:
                    video_path = f"demo/videos/{i}.mp4"
                    try:
                        if os.path.isfile(video_path):
                            video_preview = thumbnail(video_path)
                            st.image(video_preview, width=120, caption=f"Video {i}")
                            if st.button(f"Select Video {i}", key=f"vid_{i}"):
                                st.session_state.selected_video = video_path
                                st.session_state.selected_video_path = video_path
                                st.session_state.uploaded_video = None
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error loading example video {i}: {e}")

with tab2:
    st.header("Settings")
    
    # Model selection
    model = st.selectbox(
        "Select Model:",
        options=['vox', 'vox-adv', 'taichi', 'taichi-adv', 'nemo', 'mgif', 'fashion', 'bair'],
        index=0
    )
    
    if not model.startswith('vox'):
        st.warning("Warning: Upload your own images and videos (see README)")
    
    # Additional settings
    relative = st.checkbox("Relative keypoint displacement (Inherit object proportions from the video)", value=True)
    adapt_movement_scale = st.checkbox("Adapt movement scale (Don't touch unless you know what you are doing)", value=True)
    use_cpu = st.checkbox("Force CPU processing (use if having CUDA errors)", value=True)
    
    # Generate button
    if st.button("Generate Animation", use_container_width=True, type="primary"):
        if st.session_state.selected_image is not None and st.session_state.selected_video is not None:
            st.session_state.is_processing = True
            st.rerun()

with tab3:
    if st.session_state.output_video:
        st.header("Generated Animation")
        
        # Display output videos side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.video(st.session_state.output_video)
            st.download_button(
                label="Download Animation",
                data=open(st.session_state.output_video, "rb").read(),
                file_name="animated_output.mp4",
                mime="video/mp4"
            )
        
        with col2:
            st.video(st.session_state.selected_video)
            st.markdown("<p class='comparison-label'>Original Driving Video</p>", unsafe_allow_html=True)
            
        if st.button("Create New Animation"):
            st.session_state.output_video = None
            st.session_state.is_processing = False
            st.experimental_rerun()
    
    elif st.session_state.is_processing:
        st.header("Generating Animation...")
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Processing logic
        try:
            # Import torch to set device
            import torch
            
            # Force CPU if selected
            if use_cpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                torch.cuda.is_available = lambda: False
            
            load_checkpoints, make_animation = load_modules()
            
            # Download the model checkpoint if needed
            status_text.text("Downloading model checkpoint if needed...")
            progress_bar.progress(10)
            
            filename = model + ('' if model == 'fashion' else '-cpk') + '.pth.tar'
            if not os.path.isfile(filename):
                response = requests.get(
                    'https://github.com/graphemecluster/first-order-model-demo/releases/download/checkpoints/' + filename, 
                    stream=True
                )
                total_size = int(response.headers.get('Content-Length', 0))
                
                with open(filename, 'wb') as file:
                    if total_size > 0:
                        for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size//8192):
                            file.write(chunk)
                    else:
                        file.write(response.content)
            
            # Load the video frames
            status_text.text("Loading video frames...")
            progress_bar.progress(30)
            
            reader = imageio.get_reader(st.session_state.selected_video, mode='I', format='FFMPEG')
            fps = reader.get_meta_data()['fps']
            driving_video = []
            for frame in reader:
                driving_video.append(frame)
            
            # Load models
            status_text.text("Loading models...")
            progress_bar.progress(50)
            
            generator, kp_detector = load_checkpoints(
                config_path=f'config/{model}-256.yaml',
                checkpoint_path=filename
            )
            
            # Generate animation
            status_text.text("Generating animation... This may take several minutes...")
            progress_bar.progress(70)
            
            # Convert PIL Image to numpy array if needed
            source_image = np.array(st.session_state.selected_image)
            
            predictions = make_animation(
                skimage.transform.resize(source_image, (256, 256)),
                [skimage.transform.resize(frame, (256, 256)) for frame in driving_video],
                generator,
                kp_detector,
                relative=relative,
                adapt_movement_scale=adapt_movement_scale
            )
            
            # Save the output
            status_text.text("Saving animation...")
            progress_bar.progress(90)
            
            output_path = 'output.mp4'
            imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
            
            # Try to add audio from the original video
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_output:
                    # Use ffmpeg CLI directly if the Python package fails
                    try:
                        ffmpeg.output(
                            ffmpeg.input(output_path).video,
                            ffmpeg.input(st.session_state.selected_video).audio,
                            temp_output.name,
                            c='copy'
                        ).run(quiet=True, overwrite_output=True)
                        
                        with open(output_path, 'wb') as result:
                            temp_output.seek(0)
                            copyfileobj(temp_output, result)
                    except Exception as e:
                        st.warning(f"Could not add audio using ffmpeg-python: {e}. Trying command line approach.")
                        # Fallback to subprocess
                        subprocess.run([
                            'ffmpeg', '-i', output_path, '-i', st.session_state.selected_video,
                            '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0?', '-shortest', 
                            temp_output.name, '-y'
                        ], check=True)
                        
                        with open(output_path, 'wb') as result:
                            temp_output.seek(0)
                            copyfileobj(temp_output, result)
            except Exception as e:
                st.warning(f"Could not add audio to the output video: {e}")
            
            # Update session state
            st.session_state.output_video = output_path
            st.session_state.is_processing = False
            
            progress_bar.progress(100)
            status_text.text("Animation completed!")
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.is_processing = False
            import traceback
            st.code(traceback.format_exc())