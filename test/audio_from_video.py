import moviepy as mp

video = mp.VideoFileClip("dataset/samplevideo.mp4")
video.audio.write_audiofile("audio.wav")
video.close()

