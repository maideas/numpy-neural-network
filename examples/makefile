
# https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# https://trac.ffmpeg.org/wiki/Slideshow

# to extract frame 1 as PNG use:
# ffmpeg -i foo.webm -vframes 1 -f image2 -y foo.png

NAME = $(shell basename `pwd`)

videos:
	mkdir png
	./example.py
	cp `ls -1tr png/episode*.png | tail -n 1` `ls -1tr png/episode*.png | head -n 1`
	ffmpeg -i png/episode%04d.png -codec:v libvpx-vp9 -b:v 2M $(NAME).webm
	ffmpeg -i png/episode%04d.png -codec:v libtheora -qscale:v 3 -codec:a libvorbis -qscale:a 3 -f ogv $(NAME).ogv
	ffmpeg -i png/episode%04d.png -codec:v libx264 -vf "format=yuv420p" -crf 25 $(NAME).mp4

