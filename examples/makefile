
# https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# https://trac.ffmpeg.org/wiki/Slideshow

NAME = $(shell basename `pwd`)

mp4: clean
	mkdir png
	./example.py
	cp `ls -1tr png/episode*.png | tail -n 1` `ls -1tr png/episode*.png | head -n 1`
	ffmpeg -i png/episode%04d.png $(NAME).mp4

clean:
	-rm -rf png
	-rm -f $(NAME).mp4

