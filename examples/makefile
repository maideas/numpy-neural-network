
# https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# https://trac.ffmpeg.org/wiki/Slideshow

# to extract frame 1 as PNG use:
# ffmpeg -i foo.webm -vframes 1 -f image2 -y foo.png

NAME = $(shell basename `pwd`)

mp4: clean
	mkdir png
	./example.py
	cp `ls -1tr png/episode*.png | tail -n 1` `ls -1tr png/episode*.png | head -n 1`
	ffmpeg -i png/episode%04d.png -c:v libvpx-vp9 -b:v 2M $(NAME).webm

clean:
	-rm -rf png
	-rm -f $(NAME).webm

