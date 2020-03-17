
# https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
# https://trac.ffmpeg.org/wiki/Slideshow

# to extract frame 1 as PNG use:
# ffmpeg -i foo.webm -vframes 1 -f image2 -y foo.png

ifeq ($(NAME_POSTFIX),)
  NAME = $(shell basename `pwd`)
else
  NAME = $(shell basename `pwd`)$(NAME_POSTFIX)
endif

PNG_DIR = png$(NAME_POSTFIX)

ASSETS_DIR = ../../../maideas.github.io/assets/videos/

FIRST_PNG = $(shell ls -1 $(PNG_DIR)/episode*.png | sort -n | head -n 1)
LAST_PNG  = $(shell ls -1 $(PNG_DIR)/episode*.png | sort -n | tail -n 1)

WEBM_FLAGS = -codec:v libvpx-vp9 $(SIZE_FLAGS) -b:v 2M
OGV_FLAGS  = -codec:v libtheora  $(SIZE_FLAGS) -qscale:v 3 -codec:a libvorbis -qscale:a 3 -f ogv
MP4_FLAGS  = -codec:v libx264    $(SIZE_FLAGS) -crf 25

all: png_dir example encode move

encode: image1 webm ogv mp4

example:
	./example.py

png_dir:
	-rm -rf $(PNG_DIR)
	mkdir $(PNG_DIR)

image1:
	cp $(LAST_PNG) $(FIRST_PNG)
	cp $(LAST_PNG) $(NAME).png

webm:
	ffmpeg -i $(PNG_DIR)/episode%04d.png $(WEBM_FLAGS) $(NAME).webm

ogv:
	ffmpeg -i $(PNG_DIR)/episode%04d.png $(OGV_FLAGS)  $(NAME).ogv

mp4:
	ffmpeg -i $(PNG_DIR)/episode%04d.png $(MP4_FLAGS)  $(NAME).mp4

move:
	mv $(NAME).webm $(ASSETS_DIR)
	mv $(NAME).ogv  $(ASSETS_DIR)
	mv $(NAME).mp4  $(ASSETS_DIR)
	mv $(NAME).png  $(ASSETS_DIR)

