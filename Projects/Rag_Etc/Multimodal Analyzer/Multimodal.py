#!/usr/bin/env python3
"""
Gemini Media Analyzer - A clean implementation for analyzing audio and video files using Gemini models.
"""

import os
import time
import base64
import requests
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from agno.agent import Agent, RunResponse
from agno.media import Audio, Video, Image
from agno.models.google import Gemini
from agno.utils.log import logger

# Configuration
@dataclass
class Config:
    """Configuration settings for the media analyzer."""
    model_id: str = "gemini-2.0-flash-exp"
    audio_model_id: str = "gemini-2.0-flash" 
    video_model_id: str = "gemini-2.0-flash"
    default_audio_url: str = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
    default_video_path: str = "GreatRedSpot.mp4"
    markdown_output: bool = True
    stream_responses: bool = True

class MediaAnalyzer:
    """A clean media analyzer using Gemini models."""
    
    def __init__(self, config: Config = None):
        """Initialize the MediaAnalyzer with configuration."""
        self.config = config or Config()
        self._setup_environment()
        
    def _setup_environment(self):
        """Set up the working environment."""
        prog_path = Path(__file__).parent.absolute()
        os.chdir(prog_path)
        os.system('clear')
        logger.info(f"Working directory: {prog_path}")
        
    def _create_agent(self, model_id: str = None) -> Agent:
        """Create and return a configured Gemini agent."""
        model_id = model_id or self.config.model_id
        model = Gemini(id=model_id)
        
        return Agent(
            model=model,
            markdown=self.config.markdown_output,
        )
    
    def analyze_audio_from_url(self, 
                              url: str = None, 
                              prompt: str = "What is in this audio?") -> Optional[RunResponse]:
        """
        Analyze audio content from a URL.
        
        Args:
            url: URL to fetch audio from (uses default if None)
            prompt: Question to ask about the audio
            
        Returns:
            Agent response or None if failed
        """
        url = url or self.config.default_audio_url
        
        try:
            logger.info(f"Fetching audio from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            audio_data = response.content
            logger.info(f"Audio data fetched successfully ({len(audio_data)} bytes)")
            
            # Create agent for audio analysis
            agent = self._create_agent(self.config.audio_model_id)
            
            # Analyze the audio
            logger.info("Analyzing audio content...")
            return agent.run(
                prompt, 
                audio=[Audio(content=audio_data, format="wav")]
            )
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch audio from URL: {e}")
            return None
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return None
    
    def analyze_audio_from_file(self, 
                               file_path: Union[str, Path], 
                               prompt: str = "What is in this audio?") -> Optional[RunResponse]:
        """
        Analyze audio content from a local file.
        
        Args:
            file_path: Path to the audio file
            prompt: Question to ask about the audio
            
        Returns:
            Agent response or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Audio file not found: {file_path}")
            return None
            
        try:
            logger.info(f"Reading audio file: {file_path}")
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # Determine format from file extension
            audio_format = file_path.suffix.lstrip('.').lower()
            if audio_format not in ['wav', 'mp3', 'flac', 'aac']:
                audio_format = 'wav'  # Default fallback
            
            # Create agent for audio analysis
            agent = self._create_agent(self.config.audio_model_id)
            
            # Analyze the audio
            logger.info("Analyzing audio content...")
            return agent.run(
                prompt,
                audio=[Audio(content=audio_data, format=audio_format)]
            )
            
        except Exception as e:
            logger.error(f"Error analyzing audio file: {e}")
            return None
    
    def _upload_video_if_not_exists(self, client, video_path: Path) -> Optional[object]:
        """
        Upload a video to the remote server if it doesn't already exist.
        
        Args:
            client: The Gemini client instance
            video_path: Path to the video file
            
        Returns:
            Uploaded video file object or None if failed
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
            
        remote_file_name = f"files/{video_path.stem.lower().replace('_', '')}"
        
        try:
            logger.info(f"Checking for existing remote file: {remote_file_name}")
            # Try to get existing file
            video_file = client.files.get(name=remote_file_name)
            logger.info(f"Found existing file: {remote_file_name}")
            
        except Exception:
            # File doesn't exist, upload it
            logger.info(f"Uploading video: {video_path}")
            try:
                video_file = client.files.upload(
                    file=str(video_path),
                    config=dict(
                        name=remote_file_name, 
                        display_name=video_path.stem
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to upload video: {e}")
                return None
        
        # Wait for processing to complete
        max_wait_time = 300  # 5 minutes
        wait_time = 0
        
        while hasattr(video_file, 'state') and video_file.state.name == "PROCESSING":
            if wait_time >= max_wait_time:
                logger.error("Video processing timed out")
                return None
                
            logger.info("Video is processing...")
            time.sleep(2)
            wait_time += 2
            
            try:
                video_file = client.files.get(name=video_file.name)
            except Exception as e:
                logger.error(f"Error checking video status: {e}")
                return None
        
        logger.info(f"Video is ready: {video_file.name}")
        return video_file
    
    def analyze_video_from_file(self, 
                               file_path: Union[str, Path] = None, 
                               prompt: str = "Tell me about this video") -> Optional[RunResponse]:
        """
        Analyze video content from a local file.
        
        Args:
            file_path: Path to the video file (uses default if None)
            prompt: Question to ask about the video
            
        Returns:
            Agent response or None if failed
        """
        if file_path is None:
            file_path = Path(self.config.default_video_path)
        else:
            file_path = Path(file_path)
            
        if not file_path.exists():
            logger.error(f"Video file not found: {file_path}")
            return None
        
        try:
            # Create agent for video analysis
            model = Gemini(id=self.config.video_model_id)
            agent = Agent(
                model=model,
                markdown=self.config.markdown_output,
            )
            
            # Get Gemini client for file upload
            gemini_client = model.get_client()
            
            # Upload video
            video_file = self._upload_video_if_not_exists(gemini_client, file_path)
            if not video_file:
                return None
            
            # Analyze the video
            logger.info("Analyzing video content...")
            return agent.run(
                prompt,
                videos=[Video(content=video_file)],
                #stream=self.config.stream_responses,
            )
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return None

def main():
    """Main function demonstrating the MediaAnalyzer usage."""
    
    # Initialize the analyzer
    config = Config(
        model_id="gemini-2.0-flash-exp",
        markdown_output=True,
        stream_responses=True
    )
    
    analyzer = MediaAnalyzer(config)
    
    print("🎬 Gemini Media Analyzer")
    print("=" * 50)
    
    # Analyze audio from URL
    print("\n🔊 Analyzing audio from URL...")
    audio_result = analyzer.analyze_audio_from_url(
        prompt="What is in this audio? Describe what you hear."
    )
    if audio_result:
        print("\n🔊 Audio Analysis Result:")
        print(audio_result.content)

    # Analyze video from file (if exists)
    print("\n📹 Analyzing video from file...")
    video_result = analyzer.analyze_video_from_file(
        prompt="Tell me about this video. What do you see?"
    )
    if video_result:
        print("\n📹 Video Analysis Result:")
        print(video_result.content)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Analysis Summary:")
    print(f"🔊 Audio analysis: {'✅ Success' if audio_result else '❌ Failed'}")
    print(f"📹 Video analysis: {'✅ Success' if video_result else '❌ Failed'}")

def demo_audio_only():
    """Demo function focusing only on audio analysis."""
    print("🔊 Audio Analysis Demo")
    print("=" * 30)
    
    analyzer = MediaAnalyzer()
    
    # Test with different audio sources
    urls_to_test = [
        "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav",
        # Add more URLs here if needed
    ]
    
    for i, url in enumerate(urls_to_test, 1):
        print(f"\n🔊 Test {i}: Analyzing audio from {url}")
        result = analyzer.analyze_audio_from_url(
            url=url,
            prompt=f"Analyze this audio sample {i}. What do you hear?"
        )
        if not result:
            print(f"❌ Test {i} failed")

if __name__ == "__main__":
    # Check for required environment variables
    required_env_vars = ["GOOGLE_API_KEY"]  # Add other required vars
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the script.")
        exit(1)
    
    try:
        # Run the main demo
        main()
        
        # Uncomment to run audio-only demo
        # demo_audio_only()
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ An error occurred: {e}")