from py4j.java_gateway import JavaGateway
import numpy as np

gw = JavaGateway()
b = gw.entry_point
b.reset()

empty = gw.new_array(gw.jvm.int, 0)
for step in range(20):
    b.step(empty, 0, empty)

hist = np.array(b.getHostHistory())
print("Shape:", hist.shape)
for h in range(min(20, hist.shape[0])):
    vals = hist[h]
    mn, mx, avg = vals.min(), vals.max(), vals.mean()
    print("  Host %2d: min=%.3f max=%.3f avg=%.3f" % (h, mn, mx, avg))

last = hist[:, -1]
print()
print("Hosts util < 0.20:", np.sum(last < 0.20))
print("Hosts util 0.20-0.80:", np.sum((last >= 0.20) & (last <= 0.80)))
print("Hosts util > 0.80:", np.sum(last > 0.80))
print("Min util: %.4f at host %d" % (last.min(), last.argmin()))
print("Max util: %.4f at host %d" % (last.max(), last.argmax()))
print("Distribution:", np.histogram(last, bins=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])[0])
gw.shutdown()
