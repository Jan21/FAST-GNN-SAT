-include makeenv

PROGRAM_NAME=uprop

GENF=

CSRCS    = $(SOS) $(TPTPSA) $(TPTPS) $(wildcard *.cpp) $(wildcard ./generated/*.cpp)
COBJS    = $(CSRCS:.cpp=.o)
PROF_OBJS  = $(COBJS:.o=.gcda) $(COBJS:.o=.gcno) $(COBJS:.o=.gcov)
DEPENDS = ${COBJS:.o=.d}
LIBD =
LIBS =
CXX?=g++
CC=gcc

ifdef STATIC
	LNFLAGS+=-static
endif

CFLAGS+=-I.

ifdef PROF
	CFLAGS+=-fprofile-arcs -ftest-coverage
	CFLAGS+=-pg
	CFLAGS+=-g
	LNFLAGS+=-fprofile-arcs
	LNFLAGS+=-pg
	LNFLAGS+=-g
endif

ifdef DBG
	CFLAGS+=-O0
	CFLAGS+=-g
	CFLAGS+=-DDBG
	CFLAGS+=-fsanitize=address,undefined,bool
	LNFLAGS+=-fsanitize=address,undefined,bool
	MINISAT_DIR=./minisat/build/debug/lib
	MSAT=ld
else
	CFLAGS+=-DNDEBUG
	CFLAGS+=-O3
	MSAT=lr
	MINISAT_DIR=./minisat/build/release/lib
endif

ifdef NOO
	CFLAGS+=-O0
endif

CFLAGS += -Wall -Wextra
CFLAGS += -std=c++17
CFLAGS += -pedantic
CFLAGS += -MMD
LIBS+=-lz
CFLAGS += -I./generated

# =================== Decide which sat solver to use
USE_MINISAT=1

ifdef USE_EXTERNAL_SOLVER
	undefine USE_MINISAT
	CFLAGS+=-DUSE_EXTERNAL_SOLVER
endif

########### CADICAL
ifdef USE_IPASIR
	undefine USE_MINISAT
	CFLAGS+=-DUSE_IPASIR
	LIBS+=-lcadical
	CFLAGS+=-I./cadical
	LIBD+=-L./cadical
	CFLAGS+=-I./minisat
endif

########### MINISAT
ifdef USE_MINISAT
	LIBD+=-L$(MINISAT_DIR)
	MINISAT_LIB=$(MINISAT_DIR)/libminisat.a
	LIBS+=-lminisat
	CFLAGS+=-I./minisat
	CFLAGS+=-DUSE_MINISAT
endif

$(info USE_MINISAT : $(USE_MINISAT))
$(info USE_IPASIR : $(USE_IPASIR))

.PHONY: sources all make_defs

ifdef USE_MINISAT
# build minisat
all: $(MINISAT_LIB)
endif

all:
	$(MAKE) sources
	$(MAKE) $(PROGRAM_NAME)

make_defs:
	@./update_defs.sh

$(PROGRAM_NAME):  $(COBJS)
	@echo Linking: $@
	$(CXX) -o $@ $(COBJS) $(LNFLAGS) $(LIBD) $(LIBS)

$(MINISAT_LIB):
	@cd ./minisat ; $(MAKE) CXX=$(CXX) $(MSAT)

sources: $(GENF)

%.o:	%.cpp
	@echo Compiling: $@
	@$(CXX) $(CFLAGS) -c -o $@ $<

%.o:	%.c makefile
	@echo Compiling: $@
	@$(CXX) $(CFLAGS) -c -o $@ $<

clean:
	@rm -f $(DEPENDS)
	@rm -f $(GENF)
	@rm -f $(PROF_OBJS)
	@rm -f $(PROGRAM_NAME) $(PROGRAM_NAME).exe $(COBJS) *.tab.[ch]
	cd ./minisat/; $(MAKE) clean

-include ${DEPENDS}
