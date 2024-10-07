import os
from googleapiclient.discovery import build
from transformers import pipeline

# YouTube API setup
YOUTUBE_API_KEY = 'YOUR_YOUTUBE_API_KEY'  # Replace with your YouTube API Key
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Function to fetch comments from a YouTube video
def fetch_youtube_comments(video_url):
    # Extract the video ID from the URL
    video_id = video_url.split('v=')[-1]

    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100  # Max limit is 100 per page
    )
    
    while request:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
            comments.append(comment)

        # Check if more pages exist
        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100
            )
        else:
            break
    
    return comments

# Function to summarize comments using Hugging Face Transformers
def summarize_comments(comments):
    summarizer = pipeline('summarization')
    
    # Join all comments into one big text
    text = " ".join(comments)
    
    # Hugging Face has a max token limit; we need to chunk text
    max_chunk_size = 1024  # The summarizer works best with texts up to 1024 tokens
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    # Summarize each chunk
    summary = ""
    for chunk in chunks:
        summary += summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    
    return summary

# Main function
def summarize_youtube_video_comments(video_url):
    comments = fetch_youtube_comments(video_url)
    if comments:
        summary = summarize_comments(comments)
        print("Summary of YouTube comments: ")
        print(summary)
    else:
        print("No comments found.")

# Example usage
youtube_video_url = 'https://www.youtube.com/watch?v=abcde12345'  # Replace with the actual video URL
summarize_youtube_video_comments(youtube_video_url)
