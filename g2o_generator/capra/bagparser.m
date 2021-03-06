%% INIT %%
clear
format long g

% Initialising rosbag
bag = rosbag('data/raw_data/rosbags/2021-12-17-17-29-07.bag');

DECIMALS = 3;

% Odo
bSelXYT = select(bag,'Topic','/capra/robot/odometry');
msgStructsXYT = readMessages(bSelXYT,'DataFormat','struct');

xPoints = zeros(length(msgStructsXYT),1);
yPoints = zeros(length(msgStructsXYT),1);
thPoints = zeros(length(msgStructsXYT),1);
tsO = zeros(length(msgStructsXYT),1);
tsO_ = zeros(length(msgStructsXYT),1);

%% Odometry extraction 
for i=1:numel(msgStructsXYT)
   
    % Timestamp
    NanoSec = msgStructsXYT{i}.Header.Stamp.Nsec;
    tsO(i) = str2double([num2str(msgStructsXYT{i}.Header.Stamp.Sec), '.', num2str(NanoSec)]);
    tsO(i) = round(tsO(i),DECIMALS);
    
    if i > 1 && tsO(i) > tsO(i-1)
        tsO_(i) = msgStructsXYT{i}.Header.Stamp.Sec;
        
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
    end
end

% Filtering out zero elements
xPoints = nonzeros(xPoints);
yPoints = nonzeros(yPoints);
thPoints = nonzeros(thPoints);
tsO_ = nonzeros(tsO_);

%% Obect detection extraction
bSelB = select(bag,'Topic','/bearing_angles_node/output_bounding_boxes');
msgStructsB = readMessages(bSelB,'DataFormat','struct');

% Initialisation
maxyaw = 0;
tsB = zeros(length(msgStructsB),1);
tsB_ = zeros(length(msgStructsB),1);
prob_ = cell(length(msgStructsB),1);
class_ = cell(length(msgStructsB),1);
yaw_ = cell(length(msgStructsB),1);
trackID_ = cell(length(msgStructsB),1);

for i=1:numel(msgStructsB)
    % Timestamps both seconds and nanoseconds
    NanoSec = msgStructsB{i}.Header.Stamp.Nsec;
    tsB(i) = str2double([num2str(msgStructsB{i}.Header.Stamp.Sec), '.', num2str(NanoSec)]);
    tsB(i) = round(tsB(i),DECIMALS);
    
    try
        % Everytime there is an increment of time; enter the loop
        if i > 1 && tsB(i) > tsB(i-1)
            % If multiple objects in one frame or none
            n_obj = size(msgStructsB{i}.BoundingBoxes_);

            if n_obj(2) >= 1
                % Getting info bag
                prob = [msgStructsB{i}.BoundingBoxes_.Probability];
                class = [msgStructsB{i}.BoundingBoxes_.Class];
                yaw = [msgStructsB{i}.BoundingBoxes_.Yaw];
                maxyaw = max(maxyaw, yaw);
                trackID = [msgStructsB{i}.BoundingBoxes_.TrackId];
                
                % Filling cells
                tsB_(i) = msgStructsB{i}.Header.Stamp.Sec;
                prob_{i} = prob;
                class_{i} = class;
                yaw_{i} = yaw;
                trackID_{i} = trackID;
            end
        end
    catch
        % Do nothing
    end
    
end

% Filtering out zero/empty elements
tsB_ = nonzeros(tsB_);
prob_ = prob_(~cellfun('isempty',prob_));
class_ = class_(~cellfun('isempty',class_));
yaw_ = yaw_(~cellfun('isempty',yaw_));
trackID_ = trackID_(~cellfun('isempty',trackID_));

% Clearing my workspace
clear positionQuatW positionQuatX positionQuatY positionQuatZ NanoSec positionX positionY yaw ZYX trackID prob timeStamp i class n_obj

%% Synchronization
% Getting index of timestamp arrays where they are equal
[shared1, idx1] = intersect(tsO_, tsB_);
[shared2, idx2] = intersect(tsB_, tsO_);

% Extracting data
% odometry
xPoints_ = xPoints(idx1);
yPoints_ = yPoints(idx1);
thPoints_ = thPoints(idx1);

% object detection
prob_ = prob_(idx2);
class_ = class_(idx2);
yaw_ = yaw_(idx2);
trackID_ = trackID_(idx2);


%% Data visualisation

plot(xPoints,yPoints, 'LineWidth', 2, Color='red')
hold on
for i=1:numel(xPoints_)
    yaw_temp = [yaw_{i}];
    for k=1:length(yaw_temp)
        q = quiver(xPoints_(i),yPoints_(i),cos(yaw_temp(k)+thPoints_(i))*3,sin(yaw_temp(k)+thPoints_(i))*3);
        q.Color = 'blue';
        q.AutoScale = 'on';
        q.LineWidth = 2;
    end
end

legend({'Odometry','Bearing measurements'},'Location','southwest')
xlabel('x [m]') 
ylabel('y [m]') 
title('Robot route at Hasselager')
% q = quiver(xPoints_,yPoints_,cos(thPoints_),sin(thPoints_));
axis equal
grid on
hold off


%% Data export

% Extracting data
% % odometry
% xPoints_ = xPoints(idx1);
% yPoints_ = yPoints(idx1);
% thPoints_ = thPoints(idx1);
% 
% % object detection
% prob_ = prob_(idx2);
% class_ = class_(idx2);
% yaw_ = yaw_(idx2);
% trackID_ = trackID_(idx2);

% data = [xPoints, yPoints, thPoints];
% n = numel(xPoints);
% 
% class_(end+1:n) = nan;
% yaw_(end+1:n) = nan;
% trackID_(end+1:n) = nan;
% prob_(end+1:n) = nan;
% 
% data_b = [class_ yaw_, trackID_, prob_];

% data = array2table(data);
% data.Properties.VariableNames(1:7) = {'x', 'y', 'th', 'class', 'bearing', 'ID', 'prob'};
% 
% 
% data_b = array2table(data_b);
% data_b.Properties.VariableNames(1:7) = {'class', 'bearing', 'ID', 'prob'};
% 
% writetable(data, '/data/bag_data_rui.csv')
% writematrix()

