.PHONY: expose.pdf all clean

all: expose.pdf

expose.pdf: expose.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make expose.tex

clean:
	latexmk -CA
