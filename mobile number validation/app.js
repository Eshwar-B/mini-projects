$(document).ready(function() {
    $('body').on('click', '.checkmobile', function() {
        // Regular expression for 10-digit numbers
        var tdr_regex = /^\d{10}$/; 
        var mobile = $('#mobile').val();

        if (mobile !== '') {
            if (!tdr_regex.test(mobile)) {
                alert('Your phone number ' + mobile + ' is not in the correct format! It should be 10 digits.');
            } else {
                alert('Your phone number ' + mobile + ' is valid!');
            }
        } else {
            alert('You have not entered your phone number!');
        }
    });
});
