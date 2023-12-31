
import sys
from pathlib import Path
from argparse import ArgumentParser
import torch
from torch import nn
from matplotlib import pyplot as plt


class LinearRegresionModel(nn.Module):
  """Linear regression model."""
  def __init__(self):
    super().__init__()

    self.linear_layer = nn.Linear(in_features=1, out_features=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """Forward pass of the model."""
      return self.linear_layer(x)


def plot_predictions(train_data
                     ,train_labels
                     ,test_data
                     ,test_labels
                     ,predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size" : 14})


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def save_model(model):
    """Save the model state dict to a file."""

    model_path = Path("models")
    model_path.mkdir(exist_ok=True, parents=True)
    model_save_path = model_path / "linear-regression-model.pth"
    print(f"Saving model state dict at: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

def load_model(model, path):

    model.load_state_dict(torch.load(path))


class MLWorkflow:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = get_device()
        self.model_save_path = None
        self.X_train = None
        self.y_train = None
        self.X_test  = None
        self.y_test = None
        self.predictions = None
        self.loss_values = []
        self.test_loss_values = []


    def traintest_split(self, X: torch.tensor, y, train_size=0.8):
        # train/test split
        train_split = int(train_size * len(X))
        self.X_train, self.y_train = X[:train_split], y[:train_split]    
        self.X_test, self.y_test = X[train_split:], y[train_split:]

    def load_data(self):
        pass

    def plot_lossfunctions(self):
        plt.figure(figsize=(10, 7))
        plt.plot(torch.tensor(self.loss_values).numpy(), label="Train")
        plt.plot(self.test_loss_values, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def train(self, epochs=100):
        """Train the model."""
        
        for epoch in range(epochs):
            y_pred = self.model(self.X_train)
            loss = self.loss_fn(y_pred, self.y_train)
            self.loss_values.append(loss)
             # Zero gradients
            self.optimizer.zero_grad()
             # Backward pass
            loss.backward()
            # Update parameters
            self.optimizer.step()
            # Test the model
            test_loss = self.test()
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch: {epoch + 1}/{epochs}, loss: {loss.item():.3f}, test loss: {test_loss.item():.3f}")

        self.predict()

    def predict(self):
        """Predict on the test set."""
        self.model.eval()       
        with torch.inference_mode():
            self.predictions = self.model(self.X_test)
            self.plot_predictions(predictions=self.predictions)

    def test(self):
        """Test the model on the test set."""
        self.model.eval()
        with torch.inference_mode():
            test_pred = self.model(self.X_test)
            test_loss = self.loss_fn(test_pred, self.y_test)
            self.test_loss_values.append(test_loss)
            return test_loss
    

    def save_model(self, model_save_path=None):
        """Save the model state dict to a file."""
        if model_save_path is None:
            model_save_path = Path("models")
            model_save_path.mkdir(exist_ok=True, parents=True)
            model_save_path = model_save_path / "linear-regression-model.pth"
        print(f"Saving model state dict at: {model_save_path}")
        torch.save(self.model.state_dict(), model_save_path)

    def load_model(self,model_load_path=None):
        if model_load_path is None:
            model_load_path = Path("models")
            model_load_path = model_load_path / "linear-regression-model.pth"

    def plot_predictions(self, predictions=None):
        """Plot the predictions of the model on the test set."""
        
        plt.figure(figsize=(10, 7))
        plt.scatter(self.X_train, self.y_train, c="b", s=4, label="Training data")
        plt.scatter(self.X_test, self.y_test, c="g", s=4, label="Testing data")
        if predictions is not None:
            with torch.no_grad():
                plt.scatter(self.X_test, predictions, c="r", s=4, label="Predictions")
        plt.legend(prop={"size" : 14})
        plt.show()
        


def main(epochs=100, lr=0.01, step=0.02, save=False, load=False):
    """Main function."""
    weight = 0.7
    bias = 0.3
    start = 0
    end = 1    
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    torch.manual_seed(42)
    model = LinearRegresionModel()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # Instantiate the workflow
    wf = MLWorkflow(model=model, loss_fn=loss_fn, optimizer=optimizer)
    wf.traintest_split(X, y)
    wf.train(epochs=epochs)
    wf.plot_lossfunctions()
    if save:
        wf.save_model()
    

   
if __name__ == "__main__":
    a = ArgumentParser()
    a.add_argument("--epochs", default=100, type=int, help="Number of epochs")
    a.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    a.add_argument("--step", default=0.02, type=float, help="Step size")
    a.add_argument("--save", action="store_true", help="Save the model")
    a.add_argument("--load", action="store_true", help="Load the model")

    a.add_argument_group()
    args = a.parse_args()
    
    sys.exit(main( save =args.save
                  ,load =args.load
                  , epochs=args.epochs
                  , lr=args.lr
                  , step=args.step))
                   
    
    
