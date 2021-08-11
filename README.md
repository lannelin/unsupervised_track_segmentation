# unsupervised_track_segmentation

Work in progress.

Personal project - working on in spare time only.

Unsupervised segmentation of video frames with the goal of detection bounds of a go-kart track

## Acknowledgements

Building on technique described in [Kim, Kanezaki & Tanaka 2020](https://arxiv.org/abs/2007.09990).
Original implementation available [here](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip).

## TODOs

- [ ] move TODOs to issues...
- [x] lightningdataloader/dataset
  - [x] single image 
  - [x] mp4
- [x] review why crossentropy between cluster labels and features works
- [ ] **evaluation code** - want to check whether things improve
  - [ ] should be able to eval loop through to get overall score
- [ ] able to substitute MyNet for UNet
- [ ] use predictions of previous frame (or some window) to calc loss? should have similar predictions
    - probably don't want to run these in the same batch though as too similar? want to shuffle...
    - maybe run simultaneously?
- [ ] scribbles?
- [ ] validation during training?


## Notes

  
intuition behind similarity loss fn (`torch.nn.CrossEntropyLoss` between `torch.argmax(output)` and `output`):
encourages model to concretely choose a particular class for each pixel


