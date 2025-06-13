import os
import cv2
import zipfile
import tempfile
import gradio as gr
import numpy as np
import shutil
from face2face import Face2Face
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import subprocess
import ffmpeg
import torch
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import gc

# Setup directories and instantiate Face2Face once
OUTPUT_DIR, TEMP_DIR = "output", "temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
f2f = Face2Face()

# Global flags and queues
stop_processing = False
processing_queue = queue.Queue()
processing_status = {"current": 0, "total": 0, "stage": "", "substage": ""}

def update_progress(progress, current, total, stage, substage=""):
    """Update progress with detailed stage information"""
    global processing_status
    processing_status.update({
        "current": current,
        "total": total,
        "stage": stage,
        "substage": substage
    })
    progress(current / total, f"{stage}: {substage} ({current}/{total})")

def resize_for_preview(img, max_size=400):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

def enhance_face_quality(img, upscale_factor=2.0, sharpen_strength=1.2):
    """Enhance face quality through upscaling and sharpening"""
    try:
        # Convert to PIL for better quality processing
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Upscale using high-quality resampling
        orig_size = pil_img.size
        new_size = (int(orig_size[0] * upscale_factor), int(orig_size[1] * upscale_factor))
        upscaled = pil_img.resize(new_size, Image.LANCZOS)
        
        # Apply sharpening
        enhancer = ImageEnhance.Sharpness(upscaled)
        sharpened = enhancer.enhance(sharpen_strength)
        
        # Apply subtle contrast enhancement
        contrast_enhancer = ImageEnhance.Contrast(sharpened)
        enhanced = contrast_enhancer.enhance(1.1)
        
        # Convert back to CV2 format
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    except:
        return img

def preprocess_for_quality(img, target_size=1024):
    """Preprocess image for higher quality face swapping"""
    h, w = img.shape[:2]
    
    # If image is too small, upscale it
    if max(h, w) < target_size:
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return img

def cleanup_temp():
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)
    except: pass

def open_output_dir():
    """Open the output directory in the system's file explorer"""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(os.path.abspath(OUTPUT_DIR))
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['xdg-open', os.path.abspath(OUTPUT_DIR)])
    except Exception as e:
        gr.Warning(f"Could not open output directory: {str(e)}")

def process_files(src_img, tgt_files, enhance, quality_mode, upscale_factor, progress=gr.Progress()):
    global stop_processing
    stop_processing = False
    
    if src_img is None or not tgt_files:
        gr.Warning("Please upload source and target images")
        return [], None
    
    # Load target images
    tgt_imgs = []
    total_files = len(tgt_files)
    for idx, file in enumerate(tgt_files):
        try:
            img = cv2.imread(file.name)
            if img is not None:
                tgt_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            update_progress(progress, idx + 1, total_files, "Loading images", f"Loading {os.path.basename(file.name)}")
        except: pass
    
    if not tgt_imgs:
        gr.Warning("No valid target images found")
        return [], None
    
    # Process batch
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_temp = os.path.join(TEMP_DIR, f"session_{session_id}")
    os.makedirs(session_temp, exist_ok=True)
    zip_path = os.path.join(OUTPUT_DIR, f"batch_results_{session_id}.zip")
    
    output_images = []
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            total_images = len(tgt_imgs)
            for idx, tgt in enumerate(tgt_imgs):
                if stop_processing:
                    update_progress(progress, 1.0, 1.0, "Processing stopped by user")
                    return output_images, zip_path
                
                current_image = idx + 1
                update_progress(
                    progress,
                    current_image,
                    total_images,
                    "Processing images",
                    f"Image {current_image}/{total_images}"
                )
                
                # Create temp files for face2face processing
                src_temp = os.path.join(session_temp, f"src_{idx}.jpg")
                tgt_temp = os.path.join(session_temp, f"tgt_{idx}.jpg")
                
                # Preprocess for quality if enabled
                src_processed = preprocess_for_quality(cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)) if quality_mode else cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
                tgt_processed = preprocess_for_quality(cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR)) if quality_mode else cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(src_temp, src_processed)
                cv2.imwrite(tgt_temp, tgt_processed)
                
                # Face swap
                result = f2f.swap_img_to_img(src_temp, tgt_temp)
                
                # Post-process for quality enhancement
                if quality_mode:
                    result = enhance_face_quality(result, upscale_factor)
                
                if enhance:
                    result = f2f.enhance_faces(result)
                
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                
                # Save final result
                fname = f"result_{session_id}_{idx+1:03d}.jpg"
                out_path = os.path.join(OUTPUT_DIR, fname)
                cv2.imwrite(out_path, result)
                zipf.write(out_path, arcname=fname)
                
                output_images.append(resize_for_preview(result_rgb))
                
                # Clear GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        update_progress(progress, 1.0, 1.0, "Complete", "Batch processing finished!")
        return output_images, zip_path
        
    except Exception as e:
        gr.Error(f"Processing failed: {str(e)}")
        return [], None
    finally:
        shutil.rmtree(session_temp, ignore_errors=True)

def stop_processing_fn():
    global stop_processing
    stop_processing = True

# --- CORRECTED VIDEO PROCESSING FUNCTION ---
def process_video(src_img, video_path, enhance, quality_mode, upscale_factor, progress=gr.Progress()):
    """Process a video file frame by frame with enhanced progress reporting"""
    global stop_processing
    stop_processing = False
    
    if src_img is None or not video_path:
        gr.Warning("Please upload source face and target video")
        return None
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_temp = os.path.join(TEMP_DIR, f"session_{session_id}")
    os.makedirs(session_temp, exist_ok=True)
    
    src_temp = os.path.join(session_temp, "src.jpg")
    src_processed = preprocess_for_quality(cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)) if quality_mode else cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(src_temp, src_processed)
    
    temp_video_path = os.path.join(session_temp, "temp_video.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"result_{session_id}.mp4")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            gr.Error("Could not open video file")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        update_progress(progress, 0, total_frames, "Initializing", f"Video duration: {duration:.1f}s")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            gr.Error(f"Could not create temporary video writer. Your OpenCV installation may be missing video codecs.")
            cap.release()
            return None
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            if stop_processing:
                update_progress(progress, 1.0, 1.0, "Processing stopped by user")
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time
            estimated_remaining = (total_frames - frame_count) / frames_per_second if frames_per_second > 0 else 0
            
            update_progress(
                progress,
                frame_count,
                total_frames,
                "Processing video",
                f"Frame {frame_count}/{total_frames} ({frames_per_second:.1f} fps, {estimated_remaining:.1f}s remaining)"
            )
            
            tgt_temp = os.path.join(session_temp, f"tgt_frame.jpg")
            tgt_processed = preprocess_for_quality(frame) if quality_mode else frame
            cv2.imwrite(tgt_temp, tgt_processed)
            
            result = f2f.swap_img_to_img(src_temp, tgt_temp)
            
            if quality_mode:
                result = enhance_face_quality(result, upscale_factor)
            
            if enhance:
                result = f2f.enhance_faces(result)
            
            final_frame = cv2.resize(result, (width, height))
            out.write(final_frame)
            
            # Clear GPU memory periodically
            if frame_count % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
        cap.release()
        out.release()
        
        update_progress(progress, 1.0, 1.0, "Finalizing", "Combining video and audio...")
        
        try:
            input_video = ffmpeg.input(temp_video_path)
            input_audio = ffmpeg.input(video_path)
            
            stream = ffmpeg.output(
                input_video.video,
                input_audio.audio,
                output_path, 
                vcodec='libx264',
                acodec='aac',
                loglevel="error",
                shortest=None
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
        except ImportError:
            gr.Warning("`ffmpeg-python` not found. The output video will have no audio. Please run `pip install ffmpeg-python`.")
            shutil.copy2(temp_video_path, output_path)
        except Exception as e:
            gr.Warning(f"Could not combine audio (the original video may not have an audio track). The output will be silent. Reason: {str(e)}")
            shutil.copy2(temp_video_path, output_path)

        update_progress(progress, 1.0, 1.0, "Complete", "Video processing finished!")
        return output_path
        
    except Exception as e:
        gr.Error(f"Video processing failed: {str(e)}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        return None
    finally:
        shutil.rmtree(session_temp, ignore_errors=True)

def process_batch_videos(src_img, video_files, enhance, quality_mode, upscale_factor, progress=gr.Progress()):
    """Process multiple video files with enhanced progress reporting"""
    global stop_processing
    stop_processing = False
    
    if src_img is None or not video_files:
        gr.Warning("Please upload source face and target videos")
        return [], None
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_temp = os.path.join(TEMP_DIR, f"session_{session_id}")
    os.makedirs(session_temp, exist_ok=True)
    
    output_videos = []
    zip_path = os.path.join(OUTPUT_DIR, f"batch_videos_{session_id}.zip")
    
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            total_videos = len(video_files)
            for idx, video_file in enumerate(video_files):
                if stop_processing:
                    update_progress(progress, 1.0, 1.0, "Processing stopped by user")
                    return output_videos, zip_path
                
                current_video = idx + 1
                update_progress(
                    progress,
                    current_video,
                    total_videos,
                    "Processing videos",
                    f"Video {current_video}/{total_videos}: {os.path.basename(video_file.name)}"
                )
                
                # Process each video
                result_path = process_video(
                    src_img, 
                    video_file.name, 
                    enhance, 
                    quality_mode, 
                    upscale_factor,
                    progress=progress
                )
                
                if result_path:
                    output_videos.append(result_path)
                    # Add to zip file
                    zipf.write(result_path, arcname=os.path.basename(result_path))
                
                # Clear GPU memory between videos
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        update_progress(progress, 1.0, 1.0, "Complete", "Batch video processing finished!")
        return output_videos, zip_path
        
    except Exception as e:
        gr.Error(f"Batch video processing failed: {str(e)}")
        return [], None
    finally:
        shutil.rmtree(session_temp, ignore_errors=True)

# Gradio App
with gr.Blocks(title="Face2Face Swapper Pro", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 Face2Face: Source to Target Face Swapper
    ### Professional-grade face swapping with batch processing capabilities
    """)
    
    with gr.Tabs():
        with gr.TabItem("🖼️ Image Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Source Face")
                    src_img = gr.Image(type="numpy", height=350, show_label=False, image_mode="RGB")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 🧑‍🤝‍🧑 Target Images")
                    tgt_imgs = gr.File(file_count="multiple", file_types=["image"], show_label=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ Processing Options")
                    with gr.Group():
                        with gr.Row():
                            with gr.Column(scale=1):
                                enhance_chk = gr.Checkbox(
                                    label="✨ Enhance Output Faces", 
                                    value=True,
                                    info="Applies additional face enhancement to improve the final result"
                                )
                            
                            with gr.Column(scale=1):
                                quality_chk = gr.Checkbox(
                                    label="🔍 High Quality Mode", 
                                    value=True,
                                    info="Enables high-quality processing: upscales images to 1024px, applies sharpening and contrast enhancement"
                                )
                        
                        with gr.Row():
                            upscale_slider = gr.Slider(
                                1.0, 3.0, 
                                value=1.5, 
                                step=0.1, 
                                label="Face Upscale Factor",
                                info="How much to upscale faces in high quality mode (higher = better quality but slower)"
                            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    submit_btn = gr.Button("🔁 Start Processing", variant="primary", size="lg")
                with gr.Column(scale=1):
                    stop_btn = gr.Button("⏹️ Stop Processing", variant="stop", size="lg")
            
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 📸 Swapped Results")
                    with gr.Group():
                        result_gallery = gr.Gallery(
                            columns=4,
                            height=500,
                            object_fit="contain",
                            show_label=False,
                            elem_classes=["custom-gallery"]
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📦 Output Files")
                    with gr.Group():
                        zip_file_output = gr.File(label="Download All Results (.zip)", show_label=False)
                        open_dir_btn = gr.Button("📂 Open Output Directory", size="lg", variant="secondary")
        
        with gr.TabItem("🎥 Video Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Source Face")
                    video_src_img = gr.Image(type="numpy", height=350, show_label=False, image_mode="RGB")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 🎬 Target Video")
                    video_input = gr.Video(label="Upload Video", show_label=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ Processing Options")
                    with gr.Group():
                        with gr.Row():
                            with gr.Column(scale=1):
                                video_enhance_chk = gr.Checkbox(
                                    label="✨ Enhance Output Faces", 
                                    value=True,
                                    info="Applies additional face enhancement to improve the final result"
                                )
                            
                            with gr.Column(scale=1):
                                video_quality_chk = gr.Checkbox(
                                    label="🔍 High Quality Mode", 
                                    value=True,
                                    info="Enables high-quality processing: upscales frames to 1024px, applies sharpening and contrast enhancement"
                                )
                        
                        with gr.Row():
                            video_upscale_slider = gr.Slider(
                                1.0, 3.0, 
                                value=1.5, 
                                step=0.1, 
                                label="Face Upscale Factor",
                                info="How much to upscale faces in high quality mode (higher = better quality but slower)"
                            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    video_submit_btn = gr.Button("🔁 Start Video Processing", variant="primary", size="lg")
                with gr.Column(scale=1):
                    video_stop_btn = gr.Button("⏹️ Stop Processing", variant="stop", size="lg")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎬 Processed Video")
                    video_output = gr.Video(label="Output Video", show_label=False)
                    video_download_btn = gr.Button("📂 Open Output Directory", size="lg", variant="secondary")

        with gr.TabItem("🎬 Batch Video Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Source Face")
                    batch_video_src_img = gr.Image(type="numpy", height=350, show_label=False, image_mode="RGB")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 🎬 Target Videos")
                    batch_video_input = gr.File(file_count="multiple", file_types=["video"], show_label=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ Processing Options")
                    with gr.Group():
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_video_enhance_chk = gr.Checkbox(
                                    label="✨ Enhance Output Faces", 
                                    value=True,
                                    info="Applies additional face enhancement to improve the final result"
                                )
                            
                            with gr.Column(scale=1):
                                batch_video_quality_chk = gr.Checkbox(
                                    label="🔍 High Quality Mode", 
                                    value=True,
                                    info="Enables high-quality processing: upscales frames to 1024px, applies sharpening and contrast enhancement"
                                )
                        
                        with gr.Row():
                            batch_video_upscale_slider = gr.Slider(
                                1.0, 3.0, 
                                value=1.5, 
                                step=0.1, 
                                label="Face Upscale Factor",
                                info="How much to upscale faces in high quality mode (higher = better quality but slower)"
                            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    batch_video_submit_btn = gr.Button("🔁 Start Batch Processing", variant="primary", size="lg")
                with gr.Column(scale=1):
                    batch_video_stop_btn = gr.Button("⏹️ Stop Processing", variant="stop", size="lg")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 Output Files")
                    with gr.Group():
                        batch_video_zip_output = gr.File(label="Download All Results (.zip)", show_label=False)
                        batch_video_open_dir_btn = gr.Button("📂 Open Output Directory", size="lg", variant="secondary")

    # Add custom CSS for better scrolling and aesthetics
    gr.HTML("""
    <style>
        .custom-gallery {
            overflow-y: auto !important;
            max-height: 500px !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            background: var(--background-fill-secondary) !important;
        }
        .custom-gallery::-webkit-scrollbar {
            width: 8px !important;
        }
        .custom-gallery::-webkit-scrollbar-track {
            background: var(--background-fill-primary) !important;
            border-radius: 4px !important;
        }
        .custom-gallery::-webkit-scrollbar-thumb {
            background: var(--border-color-primary) !important;
            border-radius: 4px !important;
        }
        .custom-gallery::-webkit-scrollbar-thumb:hover {
            background: var(--border-color-accent) !important;
        }
        /* Add image preview size constraints */
        .image-container img {
            max-width: 100% !important;
            max-height: 350px !important;
            object-fit: contain !important;
        }
    </style>
    """)

    # Event handlers
    submit_btn.click(
        fn=process_files,
        inputs=[src_img, tgt_imgs, enhance_chk, quality_chk, upscale_slider],
        outputs=[result_gallery, zip_file_output]
    )
    
    stop_btn.click(
        fn=stop_processing_fn,
        inputs=None,
        outputs=None,
        queue=False
    )
    
    open_dir_btn.click(
        fn=open_output_dir,
        inputs=None,
        outputs=None
    )
    
    # Video processing event handlers
    video_submit_btn.click(
        fn=process_video,
        inputs=[video_src_img, video_input, video_enhance_chk, video_quality_chk, video_upscale_slider],
        outputs=[video_output]
    )
    
    video_stop_btn.click(
        fn=stop_processing_fn,
        inputs=None,
        outputs=None,
        queue=False
    )
    
    # Changed this button to open the directory
    video_download_btn.click(
        fn=open_output_dir,
        inputs=None,
        outputs=None
    )
    
    # Batch video processing event handlers
    batch_video_submit_btn.click(
        fn=process_batch_videos,
        inputs=[
            batch_video_src_img, 
            batch_video_input, 
            batch_video_enhance_chk, 
            batch_video_quality_chk, 
            batch_video_upscale_slider
        ],
        outputs=[batch_video_zip_output]
    )
    
    batch_video_stop_btn.click(
        fn=stop_processing_fn,
        inputs=None,
        outputs=None,
        queue=False
    )
    
    batch_video_open_dir_btn.click(
        fn=open_output_dir,
        inputs=None,
        outputs=None
    )

if __name__ == '__main__':
    cleanup_temp()
    demo.queue().launch()