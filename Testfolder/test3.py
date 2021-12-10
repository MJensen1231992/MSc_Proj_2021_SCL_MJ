
lm_status = {}
NodeTo = 0

for i in range(5):
    lm_status.update(dict([(i, False)]))


for i in range(5):
    for ID, status in lm_status.items():
        if ID == NodeTo and status == False:
            xguess = 5
            yguess = 5
            lm_status[NodeTo] = True

print(lm_status)