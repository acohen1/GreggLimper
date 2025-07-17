"""
YouTubeHandler Pipeline
===========
1. Input slice : List[str] (URLs not already claimed by GIF / image logic)
2. For each URL
    a. Call YouTube API to get video details.
    b. Build fragment:  [youtube] <title> — <description>
3. Return list[str] with one line per YouTube video.
"""

# TODO: Implement YouTubeHandler to process YouTube URLs