
SIZE_FLAGS = -vf scale=iw/2:ih/2:force_original_aspect_ratio=decrease

all:
	$(MAKE) -f ../makefile png_dir
	$(MAKE) -f ../makefile png_dir NAME_POSTFIX=_2
	$(MAKE) -f ../makefile example
	$(MAKE) -f ../makefile encode move
	$(MAKE) -f ../makefile encode move NAME_POSTFIX=_2

.EXPORT_ALL_VARIABLES:

