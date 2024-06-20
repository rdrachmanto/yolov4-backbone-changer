import threading
import subprocess
import os
import re


class CPU(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                output = subprocess.check_output([
                    'pidstat', '-p', str(os.getpid()), '1', '1'])
                cpu_ = float(output.splitlines()[-2].split()[-3])
                self._list.append(cpu_)

            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list
        except:
            self.result = 0, self._list

    def stop(self):
        self.event.set()


class Memory(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                output = subprocess.check_output([
                    'pidstat', '-p', str(os.getpid()), '1', '1', '-r'])
                mem_ = float(output.splitlines()[-2].split()[-3])
                self._list.append(mem_)

            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list
        except:
            self.result = 0, self._list

    def stop(self):
        self.event.set()


def jstat_start():
    subprocess.check_output(
        f'tegrastats --interval --start --logfile test.txt',
        shell=True)


def jstat_stop():
    subprocess.check_output(f'tegrastats --stop', shell=True)
    out = open("test.txt", 'r')
    lines = out.read().split('\n')
    entire_gpu = []
    entire_pow = []
    try:
        for line in lines:
            pattern = r"GR3D_FREQ (\d+)%"
            match = re.search(pattern, line)
            if match:
                gpu_ = match.group(1)
                entire_gpu.append(float(gpu_))

        for line in lines:
            pattern = r"VDD_IN (\d+)mW"
            match = re.search(pattern, line)
            if match:
                pow = match.group(1)
                entire_pow.append(float(pow))

        result_pow = sum(entire_pow) / len(entire_pow)
        result_gpu = sum(entire_gpu) / len(entire_gpu)
    except:
        result_pow = 0
        result_gpu = 0
        entire_gpu = entire_gpu
        entire_pow = entire_pow
        pass

    subprocess.check_output("rm test.txt", shell=True)
    return result_gpu, result_pow, entire_gpu, entire_pow
