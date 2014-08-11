all:
	g++ -O3 --std=c++11 -Wall -Wextra -I/usr/include/eigen3 src/*.cpp -o bin/sc
clean:
	rm bin/sc spirals_clustered.png data/spirals_clustered.csv
test:
	bin/sc
	@echo "Plotting data using R..."
	src/plot.R
