from PIL import Image

# Function to convert text to binary
def text_to_binary(text):
    binary_text = ''.join([format(ord(char), '08b') for char in text])
    return binary_text

# Function to convert binary to text
def binary_to_text(binary):
    chars = [chr(int(binary[i:i + 8], 2)) for i in range(0, len(binary), 8)]
    return ''.join(chars)

# Function to encode a message into an image
def encode_image(input_image_path, output_image_path, secret_message):
    # Open image and load its pixels
    image = Image.open()
    pixels = image.load()

    # Convert the secret message to binary
    binary_message = text_to_binary(secret_message) + '1111111111111110'  # Terminator to signal end of the message

    data_index = 0
    message_length = len(binary_message)

    for i in range(image.size[0]):  # Width
        for j in range(image.size[1]):  # Height
            pixel = list(pixels[i, j])

            for k in range(3):  # Modify each of the RGB channels
                if data_index < message_length:
                    # Replace the least significant bit with message data
                    pixel[k] = pixel[k] & ~1 | int(binary_message[data_index])
                    data_index += 1
            pixels[i, j] = tuple(pixel)
            
            if data_index >= message_length:
                break
        if data_index >= message_length:
            break

    # Save the encoded image
    image.save(output_image_path)
    print(f"Message encoded and saved as {output_image_path}")

# Function to decode a message from an image
def decode_image(image_path):
    # Open image and load its pixels
    image = Image.open(image_path)
    pixels = image.load()

    binary_message = ""
    for i in range(image.size[0]):  # Width
        for j in range(image.size[1]):  # Height
            pixel = list(pixels[i, j])

            for k in range(3):  # Extract the least significant bit from each channel
                binary_message += str(pixel[k] & 1)

            # Check for terminator
            if binary_message[-16:] == '1111111111111110':  # End signal
                binary_message = binary_message[:-16]  # Remove the terminator
                return binary_to_text(binary_message)
    
    return binary_to_text(binary_message)

# Main function to take user inputs and perform encoding
def main():
    input_image_path = input("Enter the path to the input image (e.g., input_image.png): ")
    secret_message = input("Enter the secret message you want to hide: ")
    output_image_path = input("Enter the path to save the output image (e.g., output_image.png): ")

    # Encode the message into the image
    encode_image(input_image_path, output_image_path, secret_message)

    # Optionally decode the message for verification
    decoded_message = decode_image(output_image_path)
    print(f"Decoded message from the encoded image: {decoded_message}")

# Run the main function
if __name__ == "__main__":
    main()
