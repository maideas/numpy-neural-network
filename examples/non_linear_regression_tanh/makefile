
# https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# https://trac.ffmpeg.org/wiki/Slideshow

NAME = $(shell basename `pwd`)

create_video: clean
	mkdir video
	./example.py
	cp `ls -1tr video/episode*.png | tail -n 1` `ls -1tr video/episode*.png | head -n 1`
	ffmpeg -i video/episode%04d.png video/$(NAME).mp4

clean:
	-rm -rf video

