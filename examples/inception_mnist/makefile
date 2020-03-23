
all:
	$(MAKE) -f ../makefile png_dir
	$(MAKE) -f ../makefile png_dir NAME_POSTFIX=_2
	$(MAKE) -f ../makefile example
	$(MAKE) -f ../makefile encode move
	$(MAKE) -f ../makefile encode move NAME_POSTFIX=_2

encode:
	$(MAKE) -f ../makefile encode move
	$(MAKE) -f ../makefile encode move NAME_POSTFIX=_2

.EXPORT_ALL_VARIABLES:

