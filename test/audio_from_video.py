import moviepy as mp

if __name__ == "__main__":
    video = mp.VideoFileClip("dataset/samplevideo.mp4")
    video.audio.write_audiofile("audio.wav")
    video.close()

