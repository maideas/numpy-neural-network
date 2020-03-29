
SIZE_FLAGS = -vf scale=iw/2:ih/2:force_original_aspect_ratio=decrease

all:
	$(MAKE) -f ../makefile png_dir
	$(MAKE) -f ../makefile example
	$(MAKE) -f ../makefile encode move

.EXPORT_ALL_VARIABLES:

