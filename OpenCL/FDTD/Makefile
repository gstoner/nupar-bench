EXE = test
OBJ = test.o

test: $(OBJ)
	gfortran -o $@ $^ -L. -lpfdtd -lOpenCL

%.o: %.f90
	gfortran -o $@ -c $<

clean:
	rm -fr *.log *.out *.cvp *.ps
	rm -fr $(EXE) $(OBJ)

