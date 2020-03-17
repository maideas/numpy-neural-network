
# https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# https://trac.ffmpeg.org/wiki/Slideshow

# to extract frame 1 as PNG use:
# ffmpeg -i foo.webm -vframes 1 -f image2 -y foo.png

ifeq ($(NAME_POSTFIX),)
  NAME = $(shell basename `pwd`)
else
  NAME = $(shell basename `pwd`)$(NAME_POSTFIX)
endif

ifeq ($(PNG_DIR),)
  PNG_DIR = png
endif

WEBM_FLAGS = -codec:v libvpx-vp9 $(SIZE_FLAGS) -b:v 2M
OGV_FLAGS  = -codec:v libtheora  $(SIZE_FLAGS) -qscale:v 3 -codec:a libvorbis -qscale:a 3 -f ogv
MP4_FLAGS  = -codec:v libx264    $(SIZE_FLAGS) -crf 25

all: example image1 encode move

example:
	-rm -rf $(PNG_DIR)
	mkdir $(PNG_DIR)
	./example.py

image1:
	cp `ls -1 $(PNG_DIR)/episode*.png | sort -n | tail -n 1` `ls -1 $(PNG_DIR)/episode*.png | sort -n | head -n 1`
	cp `ls -1 $(PNG_DIR)/episode*.png | sort -n | tail -n 1` $(NAME).png

webm:
	ffmpeg -i $(PNG_DIR)/episode%04d.png $(WEBM_FLAGS) $(NAME).webm

ogv:
	ffmpeg -i $(PNG_DIR)/episode%04d.png $(OGV_FLAGS)  $(NAME).ogv

mp4:
	ffmpeg -i $(PNG_DIR)/episode%04d.png $(MP4_FLAGS)  $(NAME).mp4

encode: webm ogv mp4

move:
	mv $(NAME).webm ../../../maideas.github.io/assets/videos/
	mv $(NAME).ogv  ../../../maideas.github.io/assets/videos/
	mv $(NAME).mp4  ../../../maideas.github.io/assets/videos/
	mv $(NAME).png  ../../../maideas.github.io/assets/videos/

