CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -pedantic
LDLIBS ?= -lcurl
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin

.PHONY: all release install clean

all: q

q: q.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

release: q
	strip q
	ls -lhs q
	wc -l q.c

install: q
	install -d $(DESTDIR)$(BINDIR)
	install -m 0755 q $(DESTDIR)$(BINDIR)/q

clean:
	rm -f q
