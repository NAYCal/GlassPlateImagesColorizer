from matplotlib import pyplot as plt


def display_image(axis, image, title=None, cmap='gray'):
    """Display a single image on the given axis."""
    axis.imshow(image, cmap=cmap)
    axis.axis('off')
    if title:
        axis.set_title(title)


def display_row_of_images(axes, original_img, processed_imgs, original_title=None, processed_titles=None):
    """Display a row of images (original followed by processed ones) on the given axes."""
    display_image(axes[0], original_img, original_title)

    for j, processed_img in enumerate(processed_imgs):
        title = processed_titles[j] if processed_titles else None
        display_image(axes[j + 1], processed_img, title)


def display_images(original_images, processed_images, title, original_titles=None, processed_titles=None):
    """Display original images and their corresponding processed images using matplotlib."""

    num_original = len(original_images)

    # Check if processed_images is a list of lists or a single list
    if not all(isinstance(i, list) for i in processed_images):
        processed_images = [[img] for img in processed_images]

    ncols = 1 + max(len(row) for row in processed_images)
    _, axes = plt.subplots(nrows=num_original, ncols=ncols, figsize=(5 * ncols, 5 * num_original))

    # Handle the case when there's only one row of images
    if num_original == 1:
        axes = [axes]

    for i in range(num_original):
        display_row_of_images(axes[i], original_images[i], processed_images[i],
                              original_titles[i] if original_titles else None,
                              processed_titles[i] if processed_titles else None)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
