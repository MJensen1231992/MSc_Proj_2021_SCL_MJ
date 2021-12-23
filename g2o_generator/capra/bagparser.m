%% INIT %%
clear
format long

% Initialising rosbag
bag = rosbag('data/raw_data/rosbags/2021-12-17-17-29-07.bag');

% Odo
bSelXYT = select(bag,'Topic','/capra/robot/odometry');
msgStructsXYT = readMessages(bSelXYT,'DataFormat','struct');

xPoints = zeros(length(msgStructsXYT),1);
yPoints = zeros(length(msgStructsXYT),1);
thPoints = zeros(length(msgStructsXYT),1);
ts = zeros(length(msgStructsXYT),1);

% Object detection
bSelB = select(bag,'Topic','/bearing_angles_node/output_bounding_boxes');
msgStructB = readMessages(bSelB,'DataFormat','struct');

zPoints = zeros(length(msgStructB),1);
tsB = zeros(length(msgStructB),1);

%%%%%%%%
% Do data preprocessing

% Odometry extraction 
for i=1:numel(msgStructsXYT)
   
    % X Y and THETA
    positionX = msgStructsXYT{i}.Pose.Pose.Position.X;
    positionY = msgStructsXYT{i}.Pose.Pose.Position.Y;
    
    xPoints(i) = positionX;
    yPoints(i) = positionY;
    
    % Quaternions
    positionQuatX = msgStructsXYT{i}.Pose.Pose.Orientation.X;
    positionQuatY = msgStructsXYT{i}.Pose.Pose.Orientation.Y;
    positionQuatZ = msgStructsXYT{i}.Pose.Pose.Orientation.Z;
    positionQuatW = msgStructsXYT{i}.Pose.Pose.Orientation.W;
    
    ZYX = quat2eul([positionQuatX positionQuatY positionQuatZ positionQuatW]);
    thPoints(i) = ZYX(3);
    
    % Timestamp
    NanoSec = msgStructsXYT{i}.Header.Stamp.Nsec;
    ts(i) = str2double([num2str(msgStructsXYT{i}.Header.Stamp.Sec), '.', num2str(NanoSec)]);
end

% Obect detection
% for i=numel(msgStructB)
%     
%     positionX = msgStructsB{i}.BoundingBoxes_.Pose.Position.X;
%     
%     
% end


%% Data visualisation and export

% data = [ts, xPoints, yPoints, thPoints];
% data = array2table(data);
% data.Properties.VariableNames(1:4) = {'Timestamp', 'x', 'y', 'th'};
% writetable(data, '/data/bag_data.csv')
%writematrix()



plot(xPoints,yPoints, Color='red')
hold on
q = quiver(xPoints,yPoints,cos(thPoints),sin(thPoints));
q.Color = 'blue';
q.AutoScale= 'on';
axis equal
hold off


