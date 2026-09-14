.PHONY: pdf clean

pdf: getting-started.pdf

getting-started.pdf: getting-started.tex
	latexmk -xelatex -interaction=nonstopmode -halt-on-error getting-started.tex

clean:
	latexmk -C
