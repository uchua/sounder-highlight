# Sounder Transcript Highlight Prediction

## Overview
Utilizes an implementation of TextRank to extract a highlight from a given transcript along with the associated start and end timestamps from the audio.

### Pros
- Smaller, simpler, and faster than many other methods.
- Only takes between a few seconds to minutes to return a highlight.

### Cons
- TextRank is a somewhat basic means of extracting important information from text.
- This implementation of TextRank only extracts single sentences, though it should be relatively easy to rewrite the TextRank implementation accept a window size for how much it should consider and return (e.g. 3 consecutive sentences instead of 1).
- This only looks at the text context of the podcast, so you lose a lot of additional information like tone and volume from the audio that could also indicate some portion of it should be highlight.

## Requirements
A system with Docker installed

## Build
Navigate to the `highlight` directory (the one this file is in) and run:
```bash
docker-compose build
```

## Start
After building, from the `highlight` directory:
```bash
docker-compose up
```

## Use
- `GET 127.0.0.1:8000/`: Returns status for healthchecks
- `POST 127.0.0.1:8000/predict`: POSTing a transcript JSON to this endpoint returns the predicted highlight text as well as the start and end timestamps

