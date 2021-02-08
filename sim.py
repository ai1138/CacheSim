#!/usr/bin/python3
"""
CS-UY 2214
Abed Islam
Cache Simulator
"""
import collections
import re
import argparse
import copy

# Some helpful constant values that we'll be using.
Constants = namedtuple("Constants",["NUM_REGS", "MEM_SIZE", "REG_SIZE"])
constants = Constants(NUM_REGS = 8, 
                      MEM_SIZE = 2**15,
                      REG_SIZE = 2**16)
class DLL:
		class Node:
    		def __init__(self, data=None, prev=None, next=None):
            self.data = data
            self.prev = prev
            self.next = next

        def disconnect(self):
            self.data = None
            self.prev = None
            self.next = None


    def __init__(self):
        self.head = DLL.Node()
        self.tail = DLL.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def __len__(self):
        return self.size

    def isEmpty(self):
        return len(self) == 0

    def first(self):
        if(self.isEmpty()):
            raise Exception("List is empty")
        return self.head.next

    def last(self):
        if(self.isEmpty()):
            raise Exception("List is empty")
        return self.tail.prev

    def addNext(self, node, data):
        prev = node
        succ = node.next
        new_node =DLL.Node(data, prev, succ)
        prev.next = new_node
        succ.prev = new_node
        self.size += 1
        return new_node

    def addFront(self, data):
        return self.addNext(self.head, data)

    def addEnd(self, data):
        return self.addNext(self.trailer.prev, data)

    def addBehind(self, node, data):
        return self.addNext(node.prev, data)

    def delete(self, node):
        pred = node.prev
        succ = node.next
        pred.next = succ
        succ.prev = pred
        self.size -= 1
        data = node.data
        node.disconnect()
        return data

    def deleteFront(self):
        if (self.isEmpty()):
            raise Exception("List is empty")
        return self.delete(self.first())

    def deleteEnd(self):
        if (self.isEmpty()):
            raise Exception("List is empty")
        return self.delete(self.last())
class data:
	def __init__(address,tag,LRU):
		address = address
		tag = tag
		LRU = 0

class cache
	def__init__(size,assoc,blocksize):
		size = size
		assoc = assoc
		blocksize = blocksize
		cacheType = ""
		lineNumber = (size/(assoc*blocksize))
		if(assoc == 1):#direct-mapped
			cache = []
			for i in range((size/(assoc*blocksize))):
				cache[i] = -1
			cacheType = "direct-mapped"
		elif(size == (blocksize * assoc) and size > 1):#full associative 
			cache = DLL()
			cacheType = "fully associative"
		else: #n way associative 
			cache = [[] for i in  range((size/(assoc*blocksize)))]
			cacheType = "n-way associative"
		




def intepretCommands(pgrmCnt,reg,mem,cachel1,cachel2):
	loadMem = mem[pgrmCnt]
	optCode = (loadMem & 0xe000) >> 13
#handles slti
	if(optCode == 1):
		regSrc = (loadMem & 0x1c00) >> 10
		regDst = (loadMem & 0x380) >> 7
		imm = (loadMem & 0x7f)
		if (reg[regSrc] < imm):
			reg[regDst] = 1
		else:
			reg[regDst] = 1
		pgrmCnt +=1
#handles add,sub,or,and 
	elif(optCode == 0):
		regSrcA = (loadMem & 0x1c00) >> 10
		regSrcB = (loadMem & 0x380) >> 7
		regDst =  (loadMem & 0x70) >> 4
		imm = (loadMem & 0xf)
		if(imm > 64):
			imm = imm - 128
		if(imm == 0):
			reg[regDst] = reg[regSrcA] + reg[regSrcB]
			pgrmCnt +=1
		elif(imm == 1):
			reg[regDst] = reg[regSrcA] - reg[regSrcB]
			pgrmCnt +=1
		elif(imm == 2):
			reg[regDst] = reg[regSrcA] & reg[regSrcB]
			pgrmCnt +=1		
		elif(imm == 3):
			reg[regDst] = reg[regSrcA] | reg[regSrcB]
			pgrmCnt +=1
		elif(imm == 4):
			if (reg[regSrcA] < reg[regSrcB]):
				reg[regDst] = 1
			else:
				reg[regDst] = 0
			pgrmCnt +=1
				
		elif(imm == 8):
			pgrmCnt = reg[regSrcA]
	#handles j code
	elif(optCode == 2):
		imm = (loadMem & 0xfff) 
		if(pgrmCnt == imm):
			return [True,pgrmCnt]

		pgrmCnt = imm
	#handles sw
	elif(optCode == 5):
		regAddr = (loadMem & 0x1c00) >> 10
		regSrc = (loadMem & 0x380) >> 7
		imm = (loadMem & 0x7f)
		mem[reg[regAddr] + imm] = reg[regSrc]
		pgrmCnt +=1
	#handles lw
	elif(optCode == 4):
		regAddr = (loadMem & 0x1c00) >> 10
		regDst = (loadMem & 0x380) >> 7
		imm = (loadMem & 0x7f)
		reg[regDst] = mem[reg[regAddr] + imm]
		addressToCache= reg[regAddr] + imm
		if(cachel2 == None):
			if(cachel1.cacheType == "direct-mapped"):
				block = addressToCache // cachel1.blocksize
				tag = block // cachel1.lineNumber
				line = block % cachel1.lineNumber
				if(cachel1.cache[line] == tag)
					print_cache_config("L1","HIT",pgrmCnt,addressToCache,line)
				else:
					cachel1.cache[line] = 
					print_cache_config("L1","MISS",pgrmCnt,addressToCache,line)

			elif(cachel1.cacheType == "fully associative"):

			elif(cacheL1.cacheType == "n-way associative"):

		pgrmCnt +=1
	#handles jal command
	elif(optCode == 3):
		imm = (loadMem & 0xfff) 
		reg[7] = pgrmCnt+1
		pgrmCnt = imm
	#handles jeq command 
	elif(optCode == 6):
		regA = (loadMem & 0x1c00) >> 10
		regB = (loadMem & 0x380) >> 7
		imm = (loadMem & 0x7f)
		if(imm > 64):
			imm = imm -128
		if(reg[regA] == reg[regB]):
			pgrmCnt = pgrmCnt + 1 + imm
		else:
			pgrmCnt +=1
	#handles addi command
	elif(optCode == 7):
		regSrc = (loadMem & 0x1c00) >> 10
		regDst = (loadMem & 0x380) >> 7
		imm = (loadMem & 0x7f)
		if(imm > 64):
			imm = imm -128
		reg[regDst] = reg[regSrc] + imm
		pgrmCnt +=1

	return [False,pgrmCnt]


def load_machine_code(machine_code, mem):
    """
    Loads an E20 machine code file into the list
    provided by mem. We assume that mem is
    large enough to hold the values in the machine
    code file.
    sig: list(str) -> list(int) -> NoneType
    """
    machine_code_re = re.compile("^ram\[(\d+)\] = 16'b(\d+);.*$")
    expectedaddr = 0
    for line in machine_code:
        match = machine_code_re.match(line)
        if not match:
            raise Exception("Can't parse line: %s" % line)
        addr, instr = match.groups()
        addr = int(addr,10)
        instr = int(instr,2)
        if addr != expectedaddr:
            raise Exception("Memory addresses encountered out of sequence: %s" % addr)
        expectedaddr += 1
        mem[addr] = instr

def print_state(pc, regs, memory, memquantity):
    """
    Prints the current state of the simulator, including
    the current program counter, the current register values,
    and the first memquantity elements of memory.
    sig: int -> list(int) -> list(int) - int -> NoneType
    """
    print("Final state:")
    print("\tpc="+format(pc,"5d"))
    for reg, regval in enumerate(regs):
        print(("\t$%s=" % reg)+format(regval,"5d"))
    line = ""
    for count in range(memquantity):
        line += format(memory[count], "04x")+ " "
        if count % 8 == 7:
            print(line)
            line = ""
    if line != "":
        print(line)
def print_cache_config(cache_name, size, assoc, blocksize, num_lines):
    """
    Prints out the correctly-formatted configuration of a cache.

    cache_name -- The name of the cache. "L1" or "L2"

    size -- The total size of the cache, measured in memory cells.
        Excludes metadata

    assoc -- The associativity of the cache. One of [1,2,4,8,16]

    blocksize -- The blocksize of the cache. One of [1,2,4,8,16,32,64])

    sig: str, int, int, int, int -> NoneType
    """

    summary = "Cache %s has size %s, associativity %s, " \
        "blocksize %s, lines %s" % (cache_name,
        size, assoc, blocksize, num_lines)
    print(summary)

def print_log_entry(cache_name, status, pc, addr, line):
    """
    Prints out a correctly-formatted log entry.

    cache_name -- The name of the cache where the event
        occurred. "L1" or "L2"

    status -- The kind of cache event. "SW", "HIT", or
        "MISS"

    pc -- The program counter of the memory
        access instruction

    addr -- The memory address being accessed.

    line -- The cache line or set number where the data
        is stored.

    sig: str, str, int, int, int -> NoneType
    """
    log_entry = "{event:8s} pc:{pc:5d}\taddr:{addr:5d}\t" \
        "line:{line:4d}".format(line=line, pc=pc, addr=addr,
            event = cache_name + " " + status)
    print(log_entry)

def main():
    parser = argparse.ArgumentParser(description='Simulate E20 machine')
    parser.add_argument('filename', help=
        'The file containing machine code, typically with .bin suffix')
    parser.add_argument('--cache', help=
        'Cache configuration: size,associativity,blocksize (for one cache) '
        'or size,associativity,blocksize,size,associativity,blocksize (for two caches)')
    cmdline = parser.parse_args()
    # initialize system
    pc = 0
    regs = [0] * constants.NUM_REGS
    memory = [0] * constants.MEM_SIZE

    # load program into memory
    with open(cmdline.filename) as file:
        load_machine_code(file.readlines(), memory)

    if cmdline.cache is not None:
        parts = cmdline.cache.split(",")
        if len(parts) == 3:
            [L1size, L1assoc, L1blocksize] = [int(x) for x in parts]
            stop = False
            cacheL1 = cache(L1size,L1assoc,L1blocksize)
				    while(not stop):
				    	res = intepretCommands(pc,regs,memory,cacheL1,None)
				    	pc = res[1]
				    	stop = res[0]
				    	print(pc)
				    print_state(pc, regs, memory, 128)

        elif len(parts) == 6:
            [L1size, L1assoc, L1blocksize, L2size, L2assoc, L2blocksize] = \
                [int(x) for x in parts]
            cachel1 = cache(L1size,L1assoc,L1blocksize)
            cachel2 = cache(L2size,L2assoc,L2blocksize)
            while(not stop):
				    	res = intepretCommands(pc,regs,memory,cacheL1,cachel2)
				    	pc = res[1]
				    	stop = res[0]
				    	print(pc)
				    print_state(pc, regs, memory, 128)
        else:
            raise Exception("Invalid cache config")

if __name__ == "__main__":
    main()
