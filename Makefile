CXX ?= g++
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CPPFLAGS ?=
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=

BUILD_DIR ?= build/$(UNAME_S)-$(UNAME_M)

ifeq ($(UNAME_S),Darwin)
LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(LIBOMP_PREFIX),)
CXXFLAGS += -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
LDFLAGS += -L$(LIBOMP_PREFIX)/lib
LDLIBS += -lomp
else
$(warning libomp was not found via Homebrew. Install it with: brew install libomp)
CXXFLAGS += -Xpreprocessor -fopenmp
endif
else
CXXFLAGS += -fopenmp
endif

LIB_SOURCES = code/cxx/tools.cpp code/cxx/gauss.cpp code/cxx/shell.cpp code/cxx/molecule.cpp code/cxx/dyson.cpp code/cxx/grid.cpp code/cxx/math_special.cpp code/cxx/cross_section.cpp code/cxx/angle_grid.cpp code/cxx/rotation.cpp code/cxx/continuum.cpp code/cxx/beta.cpp code/cxx/num_eikr.cpp code/cxx/point_dipole.cpp code/cxx/clebsch_gordan.cpp code/cxx/physical_dipole.cpp
LIB_OBJECTS = $(patsubst code/cxx/%.cpp,$(BUILD_DIR)/%.o,$(LIB_SOURCES))

DYSON_OBJECT = $(BUILD_DIR)/dyson_gen.o
BETA_OBJECT = $(BUILD_DIR)/beta_gen.o
INTEGRAL_OBJECT = $(BUILD_DIR)/dipole_integrals.o

EXECUTABLES = dyson_gen beta_gen dipole_integrals

all: $(EXECUTABLES)

dyson_gen: $(DYSON_OBJECT) $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

beta_gen: $(BETA_OBJECT) $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

dipole_integrals: $(INTEGRAL_OBJECT) $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: code/cxx/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build code/cxx/*.o $(EXECUTABLES)
