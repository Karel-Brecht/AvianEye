# YouTube Video Input
For youtube video input I am using the yt-dlp package. This is, up to today, very well maintained, and has lots of contributors and recent contributions.

Install yt-dlp:

```bash
pip install yt-dlp
```

I have version _2025.3.27_ installed.

# Next Steps

Potential improvements and future work to enhance capabilities and performance.

# Deployment Strategy

If I were to deploy this in a production environment.
- Containerize the service in a container with the correct environment installed.
- Load balancing: -> Kubernetes? Nginx?
- Choose appropriate video resolution to process
- Handle different resolutoins and aspect ratios
- Make logs of video's processed, video duration, nr frames, processing time
- Scalable?
- Process video in chuncks?
- Relies a lot on the yt-dlp package, there have been occurences that these kind of packages stop working due to Changes on YouTube's side. Important to use a well maintained package.
- Provide simple API
- Call it BirdsAI or BirdsAIview

Adaptations for Real-time analysis
- Process video in chuncks
- Inference must be fast, not every model can be used
- Maybe process on lower resolution and skip some frames