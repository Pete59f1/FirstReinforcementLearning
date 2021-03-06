import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

# Jeg har opstillet et "miljø" hvor agenten skal fra state 0 til state 9. I alt har vi 12 states.
# Her er en liste over de valg agenten kan tage, hvor det første tal er hvor agenter står,
# og det næste er hvor den kan gå. Kan ikke gå tilbage...

numberOfStates = 12
possibleMoves = [(0, 1), (1, 5), (5, 7), (1, 6), (6, 9), (6, 2), (2, 3), (2, 8), (8, 4), (4, 10), (1, 9), (9, 11)]
currentGoal = 9
gamma = 0.8
initialState = 0

# En graf der viser et kort over "miljøet"
G = nx.Graph()
G.add_edges_from(possibleMoves)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.show()

# Laver R vores reward matrix, og Q vores q-value matrix
# Jeg fylder R med 1 taller, og laver dem derefter om til -1
# Dette gøres for, at agenten for en straf for, at til en state med -1
R = np.matrix(np.ones([numberOfStates, numberOfStates]))
R *= -1

# Hvad der nu gøres, er at vi fjerner -1, og sætter et 0, der hvor, der er en faktisk vej, og 100 hvis det er målet.
# Med denne matrix kan agenten nu, se hvor den vil blive belønnet, og hvor den bliver straffet, afhængigt af hvilke
# Valg den tager.
for moves in possibleMoves:
    print(moves)
    if moves[1] == currentGoal:
        R[moves] = 100
    else:
        R[moves] = 0

    if moves[0] == currentGoal:
        R[moves[::-1]] = 100
    else:
        R[moves[::-1]] = 0

R[currentGoal, currentGoal] = 100

# Hvis vi udskriver R matrixen, vil y aksen vise hvilken state vores agent er ved
# Og x aksen viser hvilke valg den kan tage
# Som sagt tidligere vil -1 betyde at den ikke kan gå den state
print('')
print(R)
print('')

Q = np.matrix(np.zeros([numberOfStates, numberOfStates]))


# Skriver de metoder agenten skal bruge for at tage beslutninger og lære
# Denne metode finder hvilke muligheder agenten kan tage
def AvailableActions(state):
    currentStateRow = R[state, ]
    actions = np.where(currentStateRow >= 0)[1]
    return actions


availableActions = AvailableActions(initialState)


# Denne metode vælger en action ud fra de muligheder der er
def SampleNextAction(availableActionRange):
    nextAction = int(np.random.choice(availableActionRange, 1))
    return nextAction


action = SampleNextAction(availableActions)


# Denne metode opdaterer vores Q matrix
def Update(currentState, action, gamma):
    maxIndex = np.where(Q[action, ] == np.max(Q[action, ]))[1]

    if maxIndex.shape[0] > 1:
        maxIndex = int(np.random.choice(maxIndex, size=1))
    else:
        maxIndex = int(maxIndex)

    maxValue = Q[action, maxIndex]

    Q[currentState, action] = R[currentState, action] + gamma * maxValue

    if np.max(Q) > 0:
        return np.sum(Q/np.max(Q) * 100)
    else:
        return 0


Update(initialState, action, gamma)

# Training
scores = []
for i in range(1000):
    currentState = np.random.randint(0, int(Q.shape[0]))
    availableAction = AvailableActions(currentState)
    action = SampleNextAction(availableAction)
    score = Update(currentState, action, gamma)
    scores.append(score)
    print('Score: ', str(score))

print('')
print('Trained Q')
print(Q/np.max(Q) * 100)

# Testing
# Her kan vi ændre hvad slutmålet er, og hvor vores agent starter
# Herefter skulle den gerne komme med den bedst mulige vej
currentState = 0
steps = [currentState]

while currentState != currentGoal:
    nextStepIndex = np.where(Q[currentState, ] == np.max(Q[currentState, ]))[1]

    if nextStepIndex.shape[0] > 1:
        nextStepIndex = int(np.random.choice(nextStepIndex, size=1))
    else:
        nextStepIndex = int(nextStepIndex)

    steps.append(nextStepIndex)
    currentState = nextStepIndex

print('')
print('The best way')
print(steps)

plt.plot(scores)
plt.show()
