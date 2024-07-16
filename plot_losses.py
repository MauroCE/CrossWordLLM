import matplotlib.pyplot as plt
import pickle

# with open("losses/next_token_validation_200_10000_new.pkl", "rb") as file:
#     losses = pickle.load(file)

fig, ax = plt.subplots()
ax.plot(losses)
ax.set_xlabel("Iteration")
ax.set_ylabel("Validation Loss")
plt.show()
