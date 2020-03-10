
# https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# https://trac.ffmpeg.org/wiki/Slideshow

# to extract frame 1 as PNG use:
# ffmpeg -i foo.webm -vframes 1 -f image2 -y foo.png

NAME = $(shell basename `pwd`)

all: example image1 encode move

example:
	-rm -rf png
	mkdir png
	./example.py

image1:
	cp `ls -1tr png/episode*.png | tail -n 1` `ls -1tr png/episode*.png | head -n 1`

encode:
	ffmpeg -i png/episode%04d.png -codec:v libvpx-vp9 -b:v 2M $(NAME).webm
	ffmpeg -i png/episode%04d.png -codec:v libtheora -qscale:v 3 -codec:a libvorbis -qscale:a 3 -f ogv $(NAME).ogv
	ffmpeg -i png/episode%04d.png -codec:v libx264 -vf "format=yuv420p" -crf 25 $(NAME).mp4

move:
	mv $(NAME).webm ../../docs
	mv $(NAME).ogv  ../../docs
	mv $(NAME).mp4  ../../docs

