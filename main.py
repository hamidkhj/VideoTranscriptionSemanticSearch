from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
from typing import List, Dict, Any
import asyncio
import aiofiles
from pydantic import BaseModel
import numpy as np
import faiss
from datetime import timedelta
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse


from deepgram import DeepgramClient, PrerecordedOptions
import cohere
from moviepy.editor import VideoFileClip 
import google.generativeai as genai


app = FastAPI(title="Video Transcription Search API")

from dotenv import load_dotenv

load_dotenv()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Set your API keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
genai.configure(api_key=GEMINI_API_KEY)
deepgram = DeepgramClient(DEEPGRAM_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY)

# In-memory storage (use Redis/Database in production)
video_data = {}

class SearchRequest(BaseModel):
    video_id: str
    query: str
    top_k: int = 5

class TranscriptionChunk(BaseModel):
    text: str
    start_time: float
    end_time: float
    chunk_id: int

class SearchResult(BaseModel):
    chunk: TranscriptionChunk
    similarity_score: float

async def extract_audio_from_video_moviepy(video_path: str, audio_path: str) -> bool:
    """Extract audio from video using MoviePy"""
    print('getting the audio')
    try:
        print(f"Extracting audio from {video_path} to {audio_path}")
        
        # Use moviepy to extract audio
        def extract_audio():
            with VideoFileClip(video_path) as video:
                audio = video.audio
                audio.write_audiofile(
                    audio_path,
                    codec='pcm_s16le',
                    ffmpeg_params=['-ar', '16000', '-ac', '1'],
                    verbose=False,
                    logger=None
                )
                audio.close()
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, extract_audio)
        
        # Check if file was created
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print("Audio extraction successful")
            return True
        else:
            print("Audio extraction failed - no output file")
            return False
            
    except Exception as e:
        print(f"Error extracting audio with MoviePy: {e}")
        return False

async def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """Transcribe audio using Deepgram"""
    print('getting the transcription')
    try:
        with open(audio_path, 'rb') as audio_file:
            buffer_data = audio_file.read()

        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=True,
        )

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        return response.to_dict()
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def chunk_transcription(transcription_data: Dict[str, Any], max_chunk_duration: float = 10.0) -> List[TranscriptionChunk]:
    """Break transcription into chunks of max_chunk_duration seconds"""
    print('chunking the transcription')
    chunks = []
    
    if not transcription_data or 'results' not in transcription_data:
        return chunks
    
    # Get words with timestamps
    words = []
    for alternative in transcription_data['results']['channels'][0]['alternatives']:
        if 'words' in alternative:
            words.extend(alternative['words'])
    
    if not words:
        return chunks
    
    current_chunk_words = []
    current_chunk_start = None
    chunk_id = 0
    
    for word in words:
        word_start = word.get('start', 0)
        word_end = word.get('end', word_start)
        
        if current_chunk_start is None:
            current_chunk_start = word_start
        
        # Check if adding this word would exceed max duration
        if word_end - current_chunk_start > max_chunk_duration and current_chunk_words:
            # Create chunk from current words
            chunk_text = ' '.join([w.get('word', '') for w in current_chunk_words])
            chunk_end = current_chunk_words[-1].get('end', current_chunk_start)
            
            chunks.append(TranscriptionChunk(
                text=chunk_text,
                start_time=current_chunk_start,
                end_time=chunk_end,
                chunk_id=chunk_id
            ))
            
            # Start new chunk
            current_chunk_words = [word]
            current_chunk_start = word_start
            chunk_id += 1
        else:
            current_chunk_words.append(word)
    
    # Add final chunk
    if current_chunk_words:
        chunk_text = ' '.join([w.get('word', '') for w in current_chunk_words])
        chunk_end = current_chunk_words[-1].get('end', current_chunk_start)
        
        chunks.append(TranscriptionChunk(
            text=chunk_text,
            start_time=current_chunk_start,
            end_time=chunk_end,
            chunk_id=chunk_id
        ))
    print('chunks created!!!')
    return chunks

async def generate_summary_from_transcript(transcript: str) -> str:
    try:
        if not transcript:
            return "Transcript is empty. No summary available."


        prompt = f"""
            Summarize the following video transcription by highlighting the key points, main ideas, and important details. Provide a brief, 
            coherent summary that captures the essence of the conversation or presentation, ensuring to cover the primary topics discussed. 
            If relevant, include any conclusions or actionable items.

            Transcription: 
            {transcript}
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        return response.text.strip() if response and response.text else "No summary generated."
    except Exception as e:
        print(f"Summary generation failed: {e}")
        return "Summary generation failed."

def generate_srt_file(chunks: List[TranscriptionChunk], file_path: str) -> None:
    """Generate an SRT subtitle file from transcription chunks."""
    with open(file_path, 'w') as srt_file:
        for idx, chunk in enumerate(chunks):
            start_time = str(timedelta(seconds=chunk.start_time))
            end_time = str(timedelta(seconds=chunk.end_time))
            
            # Format the start and end times as SRT requires
            start_time = start_time.split('.')[0]
            end_time = end_time.split('.')[0]
            
            # Write the SRT content
            srt_file.write(f"{idx + 1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{chunk.text}\n\n")



def create_embeddings_and_index(chunks: List[TranscriptionChunk]) -> tuple:
    """Create embeddings using Cohere and build FAISS index"""
    print('embedding the chunks')
    try:
        texts = [chunk.text for chunk in chunks]
        
        print('getting embeddings')
        response = cohere_client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )

        print('response generate')
        
        embeddings = np.array(response.embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        print('returning embeddings')
        
        return index, embeddings
    
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None, None, None

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video file"""
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    video_id = str(uuid.uuid4())
    
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            content = await file.read()
            async with aiofiles.open(temp_video.name, 'wb') as f:
                await f.write(content)
            video_path = temp_video.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_path = temp_audio.name
        
        # Extract audio using MoviePy
        audio_extracted = await extract_audio_from_video_moviepy(video_path, audio_path)
        if not audio_extracted:
            raise HTTPException(status_code=500, detail="Failed to extract audio")
        
        # Transcribe audio
        transcription_data = await transcribe_audio(audio_path)
        if not transcription_data:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")
        
        # Extract full transcript before chunking
        alternatives = transcription_data.get("results", {}).get("channels", [])[0].get("alternatives", [])
        full_transcript = alternatives[0].get("transcript", "") if alternatives else ""
        
        summary = await generate_summary_from_transcript(full_transcript)

        # Create chunks
        chunks = chunk_transcription(transcription_data)
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to create chunks")
        
        # Generate SRT file
        srt_file_path = f"{video_id}.srt"
        generate_srt_file(chunks, srt_file_path)
        
        # Create embeddings and FAISS index
        index, embeddings = create_embeddings_and_index(chunks)
        if index is None:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
        

        
        # Store data
        video_data[video_id] = {
            'filename': file.filename,
            'chunks': chunks,
            'index': index,
            'embeddings': embeddings,
            'transcript': full_transcript, 
            'summary': summary,
            'srt_file': srt_file_path
        }
        
        # Cleanup temp files
        os.unlink(video_path)
        os.unlink(audio_path)

        print('uploading video finished successfully')

        print(len(chunks))
        print(len(video_id))
        print(len(file.filename))
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "total_chunks": len(chunks),
            'summary': summary,
            "message": "Video processed successfully",
            'srt_file': srt_file_path
        }
    
    except Exception as e:
        # Cleanup on error
        if 'video_path' in locals():
            try:
                os.unlink(video_path)
            except:
                pass
        if 'audio_path' in locals():
            try:
                os.unlink(audio_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/search/", response_model=List[SearchResult])
async def search_video(request: SearchRequest):
    """Search for similar content in the video"""
    
    if request.video_id not in video_data:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Get query embedding
        query_response = cohere_client.embed(
            texts=[request.query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        
        query_embedding = np.array(query_response.embeddings[0], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        video_info = video_data[request.video_id]
        scores, indices = video_info['index'].search(query_embedding, request.top_k)
        print(scores)
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                chunk = video_info['chunks'][idx]
                results.append(SearchResult(
                    chunk=chunk,
                    similarity_score=float(score)
                ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
@app.get("/download-srt/{video_id}")
async def download_srt(video_id: str):
    """Download the generated SRT file."""
    video = video_data.get(video_id)
    if not video or 'srt_file' not in video:
        raise HTTPException(status_code=404, detail="SRT file not found")
    
    srt_file_path = video['srt_file']
    return FileResponse(srt_file_path, media_type="application/x-subrip", filename=f"{video_id}.srt")

@app.get("/videos/")
async def list_videos():
    """List all processed videos"""
    return [
        {
            "video_id": vid_id,
            "filename": data["filename"],
            "total_chunks": len(data["chunks"])
        }
        for vid_id, data in video_data.items()
    ]

@app.get("/health/")
async def health_check():
    try:
        # Simulate a simple check, e.g., database connection check if needed
        # For now, we'll assume it's always healthy
        return JSONResponse(status_code=200, content={"status": "healthy"})
    except Exception as e:
        # In case of any failure, return 500 Internal Server Error
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)