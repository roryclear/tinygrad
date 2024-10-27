#import "ViewController.h"
#import <sys/socket.h>
#import <netinet/in.h>
#import <Foundation/Foundation.h>

@interface ViewController ()
@property (nonatomic) CFSocketRef socket;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    [self startHTTPServer];
}

- (void)startHTTPServer {
    // Create a socket
    self.socket = CFSocketCreate(NULL, PF_INET, SOCK_STREAM, IPPROTO_TCP, kCFSocketAcceptCallBack, AcceptCallback, NULL);
    if (!self.socket) {
        NSLog(@"Unable to create socket.");
        return;
    }
    
    // Set up socket address and port
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_len = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_port = htons(8081);  // Port 8080
    address.sin_addr.s_addr = INADDR_ANY;
    
    CFDataRef addressData = CFDataCreate(NULL, (const UInt8 *)&address, sizeof(address));
    if (CFSocketSetAddress(self.socket, addressData) != kCFSocketSuccess) {
        NSLog(@"Failed to bind socket to address.");
        CFRelease(self.socket);
        self.socket = NULL;
        return;
    }
    CFRelease(addressData);
    
    // Create a run loop source and add to current run loop
    CFRunLoopSourceRef source = CFSocketCreateRunLoopSource(NULL, self.socket, 0);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes);
    CFRelease(source);
    
    NSLog(@"HTTP Server started on port 8081.");
}

// Callback function to handle incoming connections
static void AcceptCallback(CFSocketRef socket, CFSocketCallBackType type, CFDataRef address, const void *data, void *info) {
    if (type != kCFSocketAcceptCallBack) return;
    
    // Accept the incoming connection
    CFSocketNativeHandle handle = *(CFSocketNativeHandle *)data;
    char buffer[1024] = {0};
    
    // Read data from the client
    ssize_t receivedBytes = recv(handle, buffer, sizeof(buffer) - 1, 0);
    if (receivedBytes < 1) {
        NSLog(@"Failed to receive data.");
        close(handle);
        return;
    }
    
    // Null-terminate and log received data
    buffer[receivedBytes] = '\0';
    NSLog(@"Received data: %s", buffer);

    // Create CFData from the received buffer
    CFDataRef dataRef = CFDataCreate(NULL, (UInt8 *)buffer, (CFIndex)receivedBytes);
    CFHTTPMessageRef httpRequest = CFHTTPMessageCreateEmpty(NULL, TRUE);
    CFHTTPMessageAppendBytes(httpRequest, CFDataGetBytePtr(dataRef), CFDataGetLength(dataRef));

    // Check if it's a complete HTTP request
    if (CFHTTPMessageIsHeaderComplete(httpRequest)) {
        // Extract the JSON body from the HTTP message
        NSData *bodyData = (__bridge_transfer NSData *)CFHTTPMessageCopyBody(httpRequest);
        
        // Parse the JSON data
        NSError *jsonError = nil;
        NSDictionary *jsonDict = [NSJSONSerialization JSONObjectWithData:bodyData options:0 error:&jsonError];
        
        if (!jsonDict || jsonError) {
            NSLog(@"Failed to parse JSON: %@", jsonError);
            close(handle);
            CFRelease(httpRequest);
            CFRelease(dataRef);
            return;
        }
        
        // Log the received JSON dictionary
        NSLog(@"Received JSON: %@", jsonDict);
        
    }
    
    // Simple response for any request
    const char *response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nHello from iPhone!";
    send(handle, response, strlen(response), 0);
    close(handle);
}

@end
