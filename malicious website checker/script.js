document.getElementById('urlForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const url = document.getElementById('urlInput').value;
    const resultDiv = document.getElementById('result');
    
    const isSafe = checkUrlSafety(url);

    if (isSafe) {
        resultDiv.textContent = 'The URL is safe to use.';
        resultDiv.style.color = 'green';
    } else {
        resultDiv.textContent = 'The URL is unsafe. Be cautious!';
        resultDiv.style.color = 'red';
    }
});

// Simulate a URL safety check
function checkUrlSafety(url) {
    try {
        const urlObj = new URL(url);

        // Check if the URL uses HTTPS
        const usesHttps = urlObj.protocol === 'https:';

        // Check for IP address in the hostname
        const ipPattern = /(\d{1,3}\.){3}\d{1,3}/;
        const hasIpAddress = ipPattern.test(urlObj.hostname);

        // Check for subdomains
        const subdomainCount = urlObj.hostname.split('.').length - 2;

        // Simulate SSL Certificate Validity check
        const sslValidity = simulateSslCheck(urlObj.hostname);

        // Simulate check for excessive pop-ups or ads (placeholder logic)
        const excessivePopups = false; // This would require behavioral analysis

        // Decide if URL is safe based on the conditions
        const isSafe = usesHttps && !hasIpAddress && subdomainCount <= 2 && sslValidity && !excessivePopups;

        console.log({
            usesHttps,
            hasIpAddress,
            subdomainCount,
            sslValidity,
            excessivePopups,
            isSafe
        });

        return isSafe;
    } catch (e) {
        console.error("Invalid URL:", e);
        return false;
    }
}

// Simulate an SSL Certificate Validity check
function simulateSslCheck(hostname) {
    // Placeholder logic for SSL check
    // In practice, use an API or backend service to verify SSL certificate validity
    return true; // Assume SSL is valid for simulation purposes
}
