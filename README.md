# Investigate a Dataset - No-show appointments
## Dataset Description
The dataset we are working is [No-show appointments data set](https://www.google.com/url?q=https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dd2e9a_noshowappointments-kagglev2-may-2016/noshowappointments-kagglev2-may-2016.csv&sa=D&ust=1532469042118000). This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment.
## Conclusions
I investigated the factors that are important for us to know in order to predict if a patient will show up for their scheduled appointment and these are the results of these analyses:

The percentage of those who did not show up is low as compared to those who showed up. 20% never showed up whereas 80% of the patients showed up for the appointment.

The factors that have a include age, neighborhood, health conditions (alcoholism, diabetes, hypertension and handicap) and scholarship.

The mean value of patient missing the appointment for the age is 19.8 and those who showed up is 80.2 indicating that most of the patients across all ages showed up for their appointments. Also, it was evident that patients aged between 15 to 35 years missed most of their appointments.

The mean value for the percentage of patients missing appointments in all neighbourhood is 80 while those showing up is 20. This indicates that patients from those from the neighbourhood have many missed appointments. Also, JARDIM CAMBURI neighbourhood had the heighest number of missed appointments.

The health conditions (alcoholism, diabetes, hypertension and handicap) is a factor to missed appointments with alcoholic patients have the highest percentage of missed appointments.

A high number of patients with scholarship missed their appointments as opposed to those without scholarship.

### Limitation
The SMS_received column was not clear on the dataset page since it was difficult to derive if it was a factor for missed appointments. This shows why receiving SMS has no effect on patients showing up for appointments from our analyses.
