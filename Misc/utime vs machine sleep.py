import machine
import utime

print('utime sleep')
old = utime.ticks_cpu()
print('sleeping for 5 sec')
utime.sleep(10)
new = utime.ticks_cpu()
print('cpu usage:', utime.ticks_diff(new,old))


print('machine sleep')
old = utime.ticks_cpu()
print('sleeping for 5 sec')
machine.sleep(10000)
new = utime.ticks_cpu()
print('cpu usage:', utime.ticks_diff(new,old))
