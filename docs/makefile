
CONFIG = --config _config.yml,_config_dev.yml
HOST   = --host localhost --port 4000

.PHONY: check clean build serve

build: clean check
	bundle exec jekyll build $(CONFIG) --incremental --verbose

serve: clean
	bundle exec jekyll serve $(CONFIG) --incremental $(HOST)

check: clean
	@echo "======== JEKYLL CHECK ========"
	bundle exec jekyll doctor $(CONFIG)
	@echo "=============================="

clean: Gemfile
	bundle exec jekyll clean
	-rm -rf .jekyll-cache
	-rm -f Gemfile.lock

Gemfile:
	-rm -f $@
	@echo 'source "https://rubygems.org"' >> $@
	@echo 'git_source(:github) {|repo_name| "https://github.com/#{repo_name}" }' >> $@
	@echo 'gem "github-pages", group: :jekyll_plugins' >> $@

