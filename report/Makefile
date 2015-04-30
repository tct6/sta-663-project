QT_TDS_report.pdf: QT_TDS_report.ipynb 
	ipython nbconvert --to=pdf QT_TDS_report.ipynb

.PHONY: all clean allclean test

all: QT_TDS_report.pdf 

clean:
	rm -rf *aux *log ../src/__pyache__/ ../src/*pyc 

allclean:
	make clean
	rm -f *pdf

test:
	py.test ~/sta-663-project/src/.

example_output.txt: example.py
	python ~/sta-663-project/example/example.py > ~/sta-663-project/example/example_output.txt
