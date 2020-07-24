# TODO

- remember how tensorflow_datasets is implemented ? maybe create a decorator such that
lets say you want to create a new dataset and add that to the tensorclan, then you just do like
```python
@tensorclan.dataset
MyDataset(BaseDataset):
    ...
```
similarly for transforms and model
```python
@tensorclan.model
MyModel():
    ...

@tensorclan.transform
MyTransforms(BaseAugmentation):
    ...
```
would be really nice, wont't it ?

- Make Trainer more generalized, or maybe such that we can use Trainer as BaseClass and just tell how
to train an epoch, test and epoch, and what all to log, and the rest of the stuff the trainer should take care
just like pytorch lightning

- Create a Package out of it once the above stuffs are done, and publish on PyPi